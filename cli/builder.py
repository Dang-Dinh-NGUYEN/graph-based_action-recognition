import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from modules import MODEL_REGISTRY
from modules import LOSS_REGISTRY
from data_generator import FEEDER_REGISTRY, DATASET_REGISTRY

def build_optimizer(optimizer_config: dict, model: nn.Module, *args):
    opt_type = optimizer_config["type"].lower()
    params_ = {k: v for k, v in optimizer_config.items() if k != "type"}

    if opt_type == "adam":
        return optim.Adam(model.parameters(), *args, **params_)
    elif opt_type == "sgd":
        return optim.SGD(model.parameters(), *args, **params_)
    elif opt_type == "adamw":
        return optim.AdamW(model.parameters(), *args, **params_)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")
    
def build_scheduler(optimizer, scheduler_config):
    if scheduler_config is None:
        return None
    
    else:
        sched_type = scheduler_config["type"].lower()
        params_ = scheduler_config.get("params", {})

        if sched_type == "steplr":
            scheduler = lr_scheduler.StepLR(
                optimizer,
                **params_
            )

        elif sched_type == "multisteplr":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                **params_
            )

        elif sched_type == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                **params_
            )

        elif sched_type == "onecycle":
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                **params_
            )

        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

        return scheduler

def build_loss_strategy(loss_config: dict):
    reduction = loss_config.get("reduction", "sum")

    if len(loss_config["list"]) == 1:
        loss_cls = LOSS_REGISTRY[loss_config["list"][0]["name"]]
        return loss_cls(**loss_config["list"][0].get("params", {}))
    else:
        return LOSS_REGISTRY["composite"](loss_config["list"], reduction=reduction)
    
def build_feeder(feeder_config: dict, **kwargs) -> nn.Module:
    feeder_type = feeder_config["type"].lower()
    params = feeder_config.get("params", {})

    if feeder_type not in FEEDER_REGISTRY:
        raise ValueError(f"Unknown dataset: {feeder_type}")
    
    return FEEDER_REGISTRY[feeder_type](**params, **kwargs)

def build_model(model_config: dict, **kwargs) -> nn.Module:
    model_type = model_config["type"].lower()
    params = model_config.get("params", {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")
    
    return MODEL_REGISTRY[model_type](**params, **kwargs)

def load_dataset(args, part="train"):
    dataset_name = args.dataset["name"].lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_REGISTRY[dataset_name]

    pre_transformer = config["pre_transform"] if args.dataset["pre_transform"] else None
    pre_filter = config["pre_filter"]

    dataset_class = config["class"]
    dataset = dataset_class(
        root=config["default_root"],
        pre_filter=pre_filter,
        pre_transform=pre_transformer,
        modality="joint",
        benchmark=args.dataset["benchmark"],
        part=part,
        extended=config["extended"]
    )
    print(f"{dataset_name} dataset loaded")
    print(f"Benchmark: {dataset.benchmark} - Modality: {args.dataset["modality"]} - Part: {part}")
    print(f"Pre_transformation: {dataset.pre_transform} - Pre_filter: {dataset.pre_filter}")
    print(f"Extended: {dataset.extended}")
    print()
    return dataset

