import yaml
from argparse import Namespace

def load_config(config_path: str) -> Namespace:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Namespace(**cfg)