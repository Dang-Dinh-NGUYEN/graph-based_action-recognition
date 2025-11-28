from collections import OrderedDict, defaultdict
import os
import random
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb
from cli.builder import build_feeder
from core.graph_based_model import graph_based_model
from cli.builder import build_loss_strategy, build_model, build_optimizer, build_scheduler
from cli.ddp_utils import cleanup_ddp, setup_ddp
from torch.utils.data import random_split, Subset
from torchinfo import summary
from core.linear import Linear
import modules
from data_generator import DATASET_REGISTRY


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_dataset(args):
    dataset_name = args.dataset["name"].lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_REGISTRY[dataset_name]

    pre_transformer = config["pre_transform"] or args.dataset.get("pre_transform", None)
    pre_filter = config["pre_filter"]

    dataset_class = config["class"]
    dataset = dataset_class(
        root=config["default_root"],
        pre_filter=pre_filter,
        pre_transform=pre_transformer,
        modality="joint",
        benchmark=args.dataset["benchmark"],
        part="train",
        extended=config["extended"]
    )

    print(f"{dataset_name} dataset loaded")
    print(f"Benchmark: {dataset.benchmark} - Modality: {args.dataset["modality"]} - Part: train")
    print(f"Pre_transformation: {dataset.pre_transform} - Pre_filter: {dataset.pre_filter}")
    print(f"Extended: {dataset.extended}")
    print()

    eval_set = dataset_class(
        root=config["default_root"],
        pre_filter=pre_filter,
        pre_transform=pre_transformer,
        modality="joint",
        benchmark=args.dataset["benchmark"],
        part="eval",
        extended=config["extended"]
    )

    print(f"{dataset_name} dataset loaded")
    print(f"Benchmark: {dataset.benchmark} - Modality: {dataset.modality} - Part: eval")
    print(f"Pre_transformation: {dataset.pre_transform} - Pre_filter: {dataset.pre_filter}")
    print(f"Extended: {dataset.extended}")
    print()

    return dataset, eval_set


def setup_data(args):
    train_set, eval_set = load_dataset(args)
    num_class = train_set.num_classes
    in_channels = train_set.num_features

    if args.debug:
        train_set = Subset(train_set, list(range(min(1000, len(train_set)))))
        eval_set = Subset(eval_set, list(range(min(1000, len(eval_set)))))
        
    return train_set, eval_set, num_class, in_channels # , val_set

def linear_ddp(rank, world_size, args):
    print(f"world size: {world_size} - current rank: {rank}")
    setup_ddp(rank, world_size)  

    try:
        # Load dataset
        train_set, eval_set, num_class, in_channels = setup_data(args)

        backbone_config = args.model["backbone"]
        classifier_config = args.model.get("classifier", None)

        # Wrap model in DistributedDataParallel
        model = graph_based_model(backbone_config=backbone_config, classifier_config=classifier_config, 
                                in_features=3, out_features=num_class)
        
        model_sd, _, _ = modules.utils.load_model(args.model.get("pre_trained", None))
        # model_sd = torch.load(args.model.get("pre_trained", None))
        cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
        
        backbone_sd = OrderedDict()
        for k, v in cleaned_sd.items():
            if k.startswith('query_enc.backbone.') and not k.startswith('query_enc.backbone.fc.') :
                backbone_sd[k[len('query_enc.backbone.'):]] = v

            elif k.startswith('encoders.motion.backbone.') and not k.startswith('encoders.motion.backbone.fc.'):
                backbone_sd[k[len('encoders.motion.backbone.'):]] = v

            elif k.startswith('encoder_q.') and k not in ['encoder_q.fc.0.weight', 'encoder_q.fc.0.bias', 'encoder_q.fc.2.weight', 'encoder_q.fc.2.bias']:
                backbone_sd[k[len('encoder_q.'):]] = v 

        print(backbone_sd.keys())
        model.backbone.load_state_dict(backbone_sd, strict=False)

        for name, param in model.backbone.named_parameters():
            # print(name)
            if not name.startswith('fc.'):
                param.requires_grad = False

        model.backbone.eval()
        for m in model.backbone.modules():
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
                
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(rank)

        if rank == 0:
            print(summary(model))
        
        train_set = build_feeder(feeder_config=args.dataset, base_dataset=train_set, cached_features=None)
        eval_set = build_feeder(feeder_config=args.dataset, base_dataset=eval_set, cached_features=None)

        # torch.cuda.empty_cache()

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        
        # print(model)

        # Initialize optimizer
        optimizer = build_optimizer(args.optimizer, model)

        # Initialize loss strategy
        loss_strategy = build_loss_strategy(args.losses)

        # Inititalize scheduler
        scheduler = build_scheduler(optimizer=optimizer, scheduler_config=getattr(args, "scheduler", None))

        # Initialize wandb writer for experiment tracking
        writer = None
        if rank == 0:
            print(f"Linear Evaluation:  {args.model["backbone"]["type"]} model on {args.dataset["name"]} dataset")
            print(f"Number of classes: {num_class}")
            print(f"Benchmark: {args.dataset["benchmark"]}")
            print(f"Modality: {args.dataset["modality"]}")
            # print(f"{len(train_set)}/{len(val_set)} as train/val split")

            # title = f"linear_{args.model["backbone"]["type"]}_{args.dataset["name"]}_{args.dataset["benchmark"]}_{args.dataset["modality"]}_{args.experiment_id}"
            title = f"linear_{args.model["backbone"]["type"]}_{args.dataset["name"]}_{args.dataset["modality"]}_{args.experiment_id}"
            
            tags = [args.model["backbone"]["type"]] + [loss["name"] for loss in args.losses["list"]] + ["linear"] + [args.dataset["name"]]

            wandb.init(
                project="graph-based_action-recognition",
                name=title,
                config=vars(args),
                group=args.model["backbone"]["type"],
                tags= tags
            )
            writer = wandb
        else:
            os.environ["WANDB_MODE"] = "disabled"
            
        trainer = Linear(model=model,
                            modality=args.dataset["modality"],
                            loss_strategy=loss_strategy,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            batch_config=args.batch_handler,
                            rank=rank,
                            world_size=world_size,
                            writer=writer,
                            autograd=args.autograd
                            )
    
        checkpoint_dir = os.path.join(
            "models",
            f"{args.model['backbone']['type']}/{args.dataset['name']}/"
            f"{args.dataset['benchmark']}/{args.dataset['modality']}/{args.experiment_id}"
        )        

        start_time = time.time()
        trainer.train(train_set, eval_set, args.n_epochs, checkpoint_dir, 5, args.from_scratch)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)

    finally:
        if rank == 0:
            print("Cleaning up ddp")
        cleanup_ddp()
        wandb.finish()
        print("All processes cleaned")