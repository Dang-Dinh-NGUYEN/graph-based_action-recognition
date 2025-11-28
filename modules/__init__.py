import os
import importlib
import pkgutil
import torch.nn as nn

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator

# Automatically import all .py modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
    