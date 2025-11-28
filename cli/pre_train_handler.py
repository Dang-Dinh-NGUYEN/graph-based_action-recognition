from collections import defaultdict
import os
import sys
import time
import torch
import wandb
from core.graph_based_model import graph_based_model
from cli.builder import load_dataset, build_loss_strategy, build_feeder, build_optimizer, build_scheduler
from cli.ddp_utils import cleanup_ddp, set_random_seed, setup_ddp
from torch.utils.data import Subset
from torchinfo import summary
from core.pre_trainer import PreTrainer
import modules
from data_generator import DATASET_REGISTRY


def setup_data(args):
    set_random_seed(args.dataset["seed"])
    dataset = load_dataset(args)
    num_class = dataset.num_classes
    in_channels = dataset.num_features

    if args.debug:
        dataset = Subset(dataset, list(range(min(1000, len(dataset)))))

    return dataset, num_class, in_channels 

def pre_train_ddp(rank, world_size, args):
    setup_ddp(rank=rank, world_size=world_size, master_port=args.port)  

    try:
        # Load dataset
        dataset, num_class, in_channels = setup_data(args)

        backbone_config = args.model["backbone"]
        classifier_config = args.model.get("classifier", None)

        # Wrap model in DistributedDataParallel
        model = graph_based_model(backbone_config=backbone_config, classifier_config=classifier_config, 
                                in_features=in_channels, out_features=num_class)
        
        if rank == 0:
            print(summary(model))
            print(model)

        train_set = build_feeder(feeder_config=args.dataset, base_dataset=dataset, cached_features=None)

        model = model.backbone.to(rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        # Initialize optimizer
        optimizer = build_optimizer(args.optimizer, model)

        # Initialize loss strategy
        loss_strategy = build_loss_strategy(args.losses)

        # Inititalize scheduler
        scheduler = build_scheduler(optimizer=optimizer, scheduler_config=getattr(args, "scheduler", None))

        # Initialize wandb writer for experiment tracking
        writer = None
        if rank == 0:
            print(f"Pre-Training {args.model["backbone"]["type"]} model on {args.dataset["name"]} dataset")
            print(f"Number of classes: {num_class}")
            print(f"Benchmark: {args.dataset["benchmark"]}")
            print(f"Modality: {args.dataset["modality"]}")

            title = f"pretext_{args.model["backbone"]["type"]}_{args.dataset["name"]}_{args.dataset["benchmark"]}_{args.dataset["modality"]}_{args.experiment_id}"
            tags = [args.model["backbone"]["type"]] + [loss["name"] for loss in args.losses["list"]] + ["pretext"] + [args.dataset["name"]]

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
            
        trainer = PreTrainer(model=model,
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
        trainer.train(train_set, args.n_epochs, checkpoint_dir, 5, args.from_scratch)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)

    finally:
        if rank == 0:
            print("Cleaning up ddp")
        cleanup_ddp()
        wandb.finish()
        print("All processes cleaned")