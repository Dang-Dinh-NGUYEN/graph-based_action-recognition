import sys
import torch
import torch.nn as nn
import torch

def save_model(output_path : str, model, optimizer, history):
    parameters = {
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'history' : history,
        }
    
    with open(output_path, "wb") as f:
        torch.save(parameters, f)

    print(f"Model saved to {output_path}", file=sys.stderr)

def load_model(model_path : str):
    with open(model_path, 'rb') as f:
        parameters = torch.load(f, weights_only=False)

    model = parameters['model']
    optimizer = parameters['optimizer']
    history = parameters['history']

    print(f"Loaded model from {model_path}", file=sys.stderr)
    return model, optimizer, history

class OutputCaptureWrapper(nn.Module):
    def __init__(self, module, name, storage_dict):
        super().__init__()
        self.module = module
        self.name = name
        self.storage_dict = storage_dict

    def forward(self, *args, **kwargs):
        # print("output capture wrapper called")
        out = self.module(*args, **kwargs)
        self.storage_dict[self.name] = out
        return out