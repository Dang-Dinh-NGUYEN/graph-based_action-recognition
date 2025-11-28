from collections import OrderedDict, defaultdict
import os
import random
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb
from cli.builder import load_dataset
from core.graph_based_model import graph_based_model
from cli.builder import build_loss_strategy, build_model, build_optimizer, build_scheduler, build_feeder
from cli.ddp_utils import cleanup_ddp, set_random_seed, setup_ddp
from core.trainer import Trainer
from data_generator.ntu_data import NTU_Dataset
from torch.utils.data import random_split, Subset
from torchinfo import summary
import modules
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from torch.utils.data import DistributedSampler, BatchSampler
from data_generator import DATASET_REGISTRY

def precompute_features(base_dataset, model, batch_size=128, num_workers=4, rank=0, world_size=1):
    model.train()
    model.to(rank)

    sampler = DistributedSampler(base_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(base_dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    local_features = {}
    with torch.no_grad():
        for batch_indices, batch in tqdm(zip(loader.batch_sampler, loader), desc=f"Precomputing features [rank {rank}]"):
            batch = batch.x.to(rank)
            batch_feats = model.extract_frozen_features(batch)
           
            for i, idx in enumerate(batch_indices):
                sample_feats = {}
                for key, value in batch_feats.items():
                    if isinstance(value, torch.Tensor):
                        sample_feats[key] = value[i].cpu()
                    elif isinstance(value, dict):
                        sample_feats[key] = {k: v[i].cpu() for k, v in value.items()}
                    else:
                        raise ValueError(f"Unknown feature type for key {key}: {type(value)}")
                local_features[idx] = sample_feats

    # Synchronize and gather from all processes
    gathered_list = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_list, local_features)

    # Merge all gathered dicts
    all_features = {}
    for d in gathered_list:
        all_features.update(d)

    return all_features

def class_balanced_split(dataset, label_ratio=0.01, seed=42, min_samples_per_class=1):
    """
    Split a dataset into labeled and unlabeled subsets, ensuring class balance in the labeled portion.
    
    Args:
        dataset (Dataset): A PyTorch dataset where `dataset[i]` returns a (data, label) tuple.
        label_ratio (float): Ratio of total data to be included in the labeled set (e.g., 0.01 = 1%).
        seed (int): Random seed for reproducibility.
        min_samples_per_class (int): Minimum number of labeled samples per class.
        
    Returns:
        labeled_subset (Subset), unlabeled_subset (Subset)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = item[1] if isinstance(item, (tuple, list)) else item.y
        label = label.item() if isinstance(label, torch.Tensor) else label
        class_to_indices[label].append(idx)

    num_classes = len(class_to_indices)
    total_samples = len(dataset)
    total_labeled = max(min_samples_per_class, int(label_ratio * total_samples))  # ensure at least one per class

    # Compute per-class budget (some may have +1 if needed to reach total_labeled)
    base_per_class = total_labeled // num_classes
    remainder = total_labeled % num_classes

    per_class_budget = {cls: base_per_class for cls in class_to_indices}
    for cls in list(per_class_budget.keys())[:remainder]:
        per_class_budget[cls] += 1

    labeled_indices = []
    unlabeled_indices = []

    for cls, indices in class_to_indices.items():
        random.shuffle(indices)
        n_labeled = min(len(indices), per_class_budget[cls])
        labeled_indices.extend(indices[:n_labeled])
        unlabeled_indices.extend(indices[n_labeled:])

    return Subset(dataset, labeled_indices), Subset(dataset, unlabeled_indices)

def setup_data(args):
    set_random_seed(args.dataset.get("seed", 42))

    dataset = load_dataset(args)
    num_class = dataset.num_classes
    in_channels = dataset.num_features

    # Split into 70% train, 30% val
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = total_len - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.dataset.get("seed", 42))
    )

    label_ratio = args.dataset.get("label_ratio", 1.0)
    
    # Use balanced splitting
    if label_ratio < 1.0:
        labeled_dataset, unlabeled_dataset = class_balanced_split(dataset, label_ratio, seed=args.dataset.get("seed", 42))
        labeled_train_set, unlabeled_train_set = class_balanced_split(train_set, label_ratio, seed=args.dataset.get("seed", 42))
        labeled_val_set, unlabeled_val_set = class_balanced_split(val_set, label_ratio, seed=args.dataset.get("seed", 42))
    else:
        labeled_dataset = dataset
        labeled_train_set = train_set
        labeled_val_set = val_set
        unlabeled_dataset = None
        unlabeled_train_set = None
        unlabeled_val_set = None

    if args.debug:
        labeled_train_set = Subset(labeled_train_set, list(range(min(1000, len(labeled_train_set)))))
        labeled_val_set = Subset(labeled_val_set, list(range(min(500, len(labeled_val_set)))))

    return labeled_dataset, labeled_train_set, labeled_val_set, unlabeled_dataset, unlabeled_train_set, unlabeled_val_set, num_class, in_channels

def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)  

    try:
        # Load dataset
        labeled_dataset, labeled_train_set, labeled_val_set, unlabeled_dataset, unlabeled_train_set, unlabeled_val_set, num_class, in_channels = setup_data(args)

        backbone_config = args.model["backbone"]
        classifier_config = args.model.get("classifier", None)

        # Wrap model in DistributedDataParallel
        model = graph_based_model(backbone_config=backbone_config, classifier_config=classifier_config, 
                                in_features=in_channels, out_features=num_class).to(rank)
        
        if args.model.get("pre_trained", None):
            model_sd, _, _ = modules.utils.load_model(args.model.get("pre_trained", None))
            cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())

            backbone_sd = OrderedDict()
            for k, v in cleaned_sd.items():
                if k.startswith('query_enc.backbone.') and not k.startswith('query_enc.backbone.fc.'):
                    backbone_sd[k[len('query_enc.backbone.'):]] = v

                elif k.startswith('encoders.motion.backbone.') and not k.startswith('encoders.motion.backbone.fc.'):
                    backbone_sd[k[len('encoders.motion.backbone.'):]] = v

            model.backbone.load_state_dict(backbone_sd, strict=False)

        if rank == 0:
            print(summary(model))

        if args.dataset.get("use_frozen_features", False):
            train_cached_features = precompute_features(base_dataset=train_set, model=model, rank=rank, world_size=world_size)
            print(train_cached_features[0])
            train_set = build_feeder(feeder_config=args.dataset, base_dataset=train_set, cached_features=train_cached_features, device=rank)
           
            val_cached_features = precompute_features(base_dataset=val_set, model=model, rank=rank, world_size=world_size)
            val_set = build_feeder(feeder_config=args.dataset, base_dataset=val_set, cached_features=val_cached_features, device=rank)
        else:
            train_set = build_feeder(feeder_config=args.dataset, base_dataset=labeled_dataset, cached_features=None)
            # val_set = build_feeder(feeder_config=args.dataset)  # , base_dataset=labeled_val_set, cached_features=None)

        torch.cuda.empty_cache()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

        # Initialize optimizer
        optimizer = build_optimizer(args.optimizer, model)

        # Initialize loss strategy
        loss_strategy = build_loss_strategy(args.losses)

        # Inititalize scheduler
        scheduler = build_scheduler(optimizer=optimizer, scheduler_config=getattr(args, "scheduler", None))

        # Initialize wandb writer for experiment tracking
        writer = None
        if rank == 0:
            print(f"Training {args.model["backbone"]["type"]} model on {args.dataset["name"]} dataset")
            print(f"Number of classes: {num_class}")
            print(f"Benchmark: {args.dataset["benchmark"]}")
            print(f"Modality: {args.dataset["modality"]}")
            print(f"{len(train_set)} labeled samples")
            # print(f"{len(train_set)}/{len(val_set)} as train/val split")

            title = f"{args.model["backbone"]["type"]}_{args.dataset["name"]}_{args.dataset["benchmark"]}_{args.dataset["modality"]}_{args.experiment_id}"
            tags = [args.model["backbone"]["type"]] + [loss["name"] for loss in args.losses["list"]]

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
            
        trainer = Trainer(model=model,
                        modality = args.dataset["modality"],
                        loss_strategy=loss_strategy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        batch_config=args.batch_handler,
                        rank=rank,
                        world_size=world_size,
                        writer=writer,
                        )
    
        parts = [
            "models",
            args.model['backbone']['type'],
            args.dataset['name'],
        ]

        if args.dataset.get('benchmark') is not None:
            parts.append(args.dataset['benchmark'])

        parts.extend([args.dataset['modality'], args.experiment_id])

        checkpoint_dir = os.path.join(*parts)   

        start_time = time.time()
        trainer.train(train_set, None, args.n_epochs, checkpoint_dir, 5, args.from_scratch)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)

    finally:
        if rank == 0:
            print("Cleaning up ddp")
        cleanup_ddp()
        wandb.finish()
        print("All processes cleaned")