from collections import OrderedDict
import torch

from cli.builder import build_feeder, load_dataset
from cli.config_handler import load_config
from core.graph_based_model import graph_based_model
from core.evaluator import *
import modules

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    feeders = []

    for config, model_path in zip(args.model_config, args.model_path):
        model_config = load_config(config)

        # Load dataset
        dataset = load_dataset(model_config, part="eval")
        num_class = dataset.num_classes
        in_channels = dataset.num_features
        
        feeder = build_feeder(feeder_config=model_config.dataset, base_dataset=dataset, cached_features=None)
        feeders.append(feeder)

        backbone_config = model_config.model["backbone"]
        classifier_config = model_config.model.get("classifier", None)

        # Wrap model in DistributedDataParallel
        model = graph_based_model(backbone_config=backbone_config, classifier_config=classifier_config, 
                                    in_features=in_channels, out_features=num_class).to(device)
        
        # print(model)

        # Load model's state dict
        model_sd, _, _ = modules.utils.load_model(model_path)
        cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
        # print(cleaned_sd.keys())
        model.load_state_dict(cleaned_sd, strict=True)
        model.eval()
        models.append(model)

    print(f"{len(models)} model.s detected")
    print(f"{len(feeders)} dataset.s detected")

    evaluator = Evaluator(models=models, datasets=feeders, modalities=args.modalities, metric=args.metric, alphas=args.alphas, batch_config={"type": "basic_batch_handler", "batch_size": 256})
    evaluator.evaluate(display=args.display)