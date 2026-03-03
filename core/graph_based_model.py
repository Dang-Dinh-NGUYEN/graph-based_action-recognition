from collections import OrderedDict
import time
import torch
import torch.nn as nn

from cli.builder import build_model
from modules.utils import load_model, OutputCaptureWrapper

class FrozenFeatureReplacer(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, *args, **kwargs):
        return self.output

def get_parent_module(model, module_name):
    parts = module_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

class graph_based_model(nn.Module):
    """
    Principal model for action recognition task
    """

    def __init__(self, backbone_config=None, classifier_config=None, in_features=3, out_features=60, pre_trained=None, **kwargs):
        super().__init__()
        backbone_sd = None
        classifier_sd = None

        if pre_trained:
            model_sd, _, _ = load_model(pre_trained)
            cleaned_model_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
            
            backbone_sd = OrderedDict()
            for k, v in cleaned_model_sd.items():
                if k.startswith("key_enc."):
                    backbone_sd[k[len("key_enc."):]] = v


            classifier_sd = OrderedDict()
            for k, v in cleaned_model_sd.items():
                if k.startswith("classifier."): # backbone.encoder.
                    classifier_sd[k[len("classifier."):]] = v
            classifier_sd = OrderedDict((k.replace("classifier.", ""), v) for k, v in classifier_sd.items())

        if backbone_config:    
            self.backbone = build_model(model_config=backbone_config, in_channels=in_features, pre_trained_sd=backbone_sd, num_class=out_features)
            if backbone_sd:
                self.backbone.load_state_dict(backbone_sd, strict=True)

        if classifier_config:
            self.classifier=build_model(model_config=classifier_config, out_features=out_features, pre_trained_sd=classifier_sd)
        self.hooks = []
        self.outputs = {}
        # Initialize outputs dict for frozen feature caching
        self.outputs = {}
        self._original_modules = {}

    def extract_features(self, x, **kwargs):
        return self.backbone(x, **kwargs)
    
    def forward(self, x, frozen_features=None,  **kwargs):
        if frozen_features is not None:
            # start_time = time.time()
            self._inject_frozen_features(frozen_features)
            # end_time = time.time()
            # print(f"Frozen features injected in {end_time - start_time:.4f} seconds\n")

        output = self.extract_features(x, frozen_features=frozen_features, **kwargs)
        # output["features"] = output["features"].view(output["features"].size(0), -1)
        logits = output["features"].view(output["features"].size(0), -1)
        # logits = self.classifier(output["features"]) if self.classifier else None
        # print(torch.isnan(output["features"]).any(), torch.isinf(output["features"]).any())
         # Restore original modules to allow future calls without frozen features
        if frozen_features is not None:
            self._restore_original_modules()

        return {"logits": logits, **output}
    
    def _wrap_frozen_modules(self, max_depth=2):
        self.outputs = {}

        def should_wrap(module):
            return (
            isinstance(module, nn.Module) and
            not isinstance(module, OutputCaptureWrapper) and
            all(not p.requires_grad for p in module.parameters())
            )

        def recurse(module, prefix="", current_depth=0):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
        
                if not isinstance(child, nn.Module):
                    continue

                if should_wrap(child) and current_depth == max_depth:
                    print(f"[FROZEN] Wrapping submodule {full_name}")
                    wrapped = OutputCaptureWrapper(child, full_name, self.outputs)
                    setattr(module, name, wrapped)
                    self._original_modules[full_name] = child
              
                recurse(child, full_name, current_depth + 1)

        recurse(self)

    def extract_frozen_features(self, x):
        if not getattr(self, "_hooks_registered", False):
            # print("Wrapping frozen modules...")
            self._wrap_frozen_modules()
            self._hooks_registered = True
            
        self.outputs.clear()
        with torch.no_grad():
            _ = self.extract_features(x)
        return  self.outputs

    def _inject_frozen_features(self, frozen_features):
        # Restore modules if previously replaced
        self._restore_original_modules()
        for k, v in frozen_features.items():
            for name, cached_out in v.items():
                # print(f"{name} {cached_out}")
                parent, child_name = get_parent_module(self, name)
                # Save original module to restore later
                if name not in self._original_modules:
                    self._original_modules[name] = getattr(parent, child_name)
                # Replace with dummy module returning cached output
                setattr(parent, child_name, FrozenFeatureReplacer(cached_out))
       
    def _restore_original_modules(self):
        for name, original_module in self._original_modules.items():
            parent, child_name = get_parent_module(self, name)
            setattr(parent, child_name, original_module)
        self._original_modules.clear()