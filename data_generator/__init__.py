from collections.abc import Mapping
import importlib
import pkgutil
import numpy as np
import torch
from data_generator.ntu_data import NTU_Dataset
from data_generator.pku_data import PKU_Dataset

BATCH_HANDLER_REGISTRY = {}
def register_batch_handler(name):
    def decorator(cls):
        BATCH_HANDLER_REGISTRY[name] = cls
        return cls
    return decorator

FEEDER_REGISTRY = {}
def register_feeder(name):
    def decorator(cls):
        FEEDER_REGISTRY[name] = cls
        return cls
    return decorator

AUGMENTATION_REGISTRY = {}
def register_augmentation(name):
    def decorator(cls):
        AUGMENTATION_REGISTRY[name] = cls
        return cls
    return decorator

DATASET_REGISTRY = {
    "nturgb+d": {
        "class": NTU_Dataset,
        "default_root": "data/nturgb+d_skeletons/",
        "pre_filter": NTU_Dataset.__nturgbd_pre_filter__,
        "pre_transform": NTU_Dataset.__nturgbd_pre_transformer__,
        "extended": False,
    },

    "nturgb+d_120": {
        "class": NTU_Dataset,
        "default_root": "data/nturgb+d_skeletons/",
        "pre_filter": NTU_Dataset.__nturgbd_pre_filter__,
        "pre_transform": NTU_Dataset.__nturgbd_pre_transformer__,
        "extended": True,
    },

    "pku_mmd": {
        "class": PKU_Dataset,
        "default_root": "data/PKUMMD/",
        "pre_filter": PKU_Dataset.__pku_pre_filter__,
        "pre_transform": PKU_Dataset.__pku_pre_transformer__,
        "extended": False,
    },

    "pku_mmd_v2": {
        "class": PKU_Dataset,
        "default_root": "data/PKUMMD/",
        "pre_filter": PKU_Dataset.__pku_pre_filter__,
        "pre_transform": PKU_Dataset.__pku_pre_transformer__,
        "extended": True,
    },     
}

def linear_interpolation_sampling(num_frames, MAX_FRAME):
    """
    Resize a sequence to MAX_FRAME using linear interpolation sampling.
    Selects frames evenly spaced between 0 and num_frames - 1.
    """
    # Generate evenly spaced positions
    selected_idx = np.linspace(0, num_frames - 1, MAX_FRAME, dtype=int)
    return selected_idx

def recursive_stack(batch_list):
        """
        Recursively stack a list of nested dictionaries of tensors.
        """
        if isinstance(batch_list[0], torch.Tensor):
            return torch.stack(batch_list)

        elif isinstance(batch_list[0], Mapping):
            return {
                key: recursive_stack([d[key] for d in batch_list])
                for key in batch_list[0]
            }

        else:
            raise TypeError(f"Unsupported type in recursive_stack: {type(batch_list[0])}")


def recursive_to_device(batch, device):
    """
    Recursively move nested dictionaries of tensors to the specified device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)

    elif isinstance(batch, Mapping):
        return {
            key: recursive_to_device(value, device)
            for key, value in batch.items()
        }

    else:
        raise TypeError(f"Unsupported type in recursive_to_device: {type(batch)}")
    
# Automatically import all .py modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")