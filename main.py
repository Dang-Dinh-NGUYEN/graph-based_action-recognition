import argparse
import os
import sys
import time
import torch
from torch import nn
from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from modules.ms_aagcn import ms_aagcn
from modules.trainer import Trainer
from modules.evaluator import *
from modules.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split
from collections import OrderedDict

# --- Process Parser ---

def handle_process_parser(args):
    print(f"Processing {os.path.basename(args.dataset.rstrip('/'))}")
    print(f"Expect {120 if args.extended else 60} classes")
    print(f"Benchmark: {args.benchmark} - Modality: {args.modality} - Part: {args.part}")

    pre_transformer = None
    if args.pre_transform:
        pre_transformer =  NTU_Dataset.__nturgbd_pre_transformer__
        print(f"Pre_transformation selected")

    dataset = NTU_Dataset(root=args.dataset,
                          pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                          pre_transform=pre_transformer,
                          modality=args.modality,
                          benchmark=args.benchmark,
                          part=args.part,
                          extended=args.extended)
    
    # TO DO : RE-DEFINE get_summary and print_summary
    if args.summary:
        dataset.print_summary()

# --- Train Parser ---

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9991'

    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def handle_train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    pre_transformer = None
    if args.pre_transform:
        pre_transformer =  NTU_Dataset.__nturgbd_pre_transformer__

    dataset = NTU_Dataset(root=args.dataset,
                          pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                          pre_transform=pre_transformer,
                          modality=args.modality,
                          benchmark=args.benchmark,
                          part="train",
                          extended=args.extended
                          )

    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = total_len - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    if rank == 0:
        print(f"Training model on {os.path.basename(args.dataset.rstrip('/'))} dataset...")
        print(f"Number of classes: {120 if args.extended else 60}")
        print(f"Benchmark: {args.benchmark}")
        print(f"Modality: {args.modality}")
        print(f"{len(train_set)}/{len(val_set)} as train/val split")

    # Create model and move to correct GPU
    num_classes = 120 if args.extended else 60
    model = ms_aagcn(num_class=num_classes).to(rank)

    # Wrap model in DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0001)


    trainer = Trainer(model=model,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        batch_size=args.batch_size,
                        rank=rank,
                        world_size=world_size,
                        device=device
                        )

    try:
        start_time = time.time()
        history = trainer.train(train_set, val_set, args.n_epochs)
        end_time = time.time()

        if rank == 0:
            print(f"Training time {end_time - start_time:.2f} seconds", file=sys.stderr)
            model_path = os.path.join("models", f"ntu_rgbd{120 if args.extended else ''}_{args.benchmark}_{args.modality}.pt")
            save_model(model_path, model, optimizer, loss_function, history, args.batch_size)

    finally:
        cleanup_ddp()

def handle_train_parser(args):
    world_size = torch.cuda.device_count()
    print(f"Prepare training process on {world_size} GPU")
    mp.spawn(handle_train_ddp, args=(world_size, args), nprocs=world_size, join=True)

# --- EVAL PARSER ---

def handle_eval_parser(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {args.benchmark} benchmark")

    dataset = NTU_Dataset(root=args.dataset,
                          pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                          pre_transform=NTU_Dataset.__nturgbd_pre_transformer__,
                          modality=args.modality,
                          benchmark=args.benchmark,
                          part="eval",
                          extended=args.extended)

    model = ms_aagcn().to(device)
    model_sd, _, _, _, batch_size = load_model(args.model)
    cleaned_sd_1 = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
    model.load_state_dict(cleaned_sd_1, strict=False)  # strict=False allows partial load

    evaluator = Evaluator(model=model, batch_size=batch_size)
    evaluation = evaluator.evaluate(dataset=dataset, topk=(1, 5), display=args.display)

    print(f"Top-1 Accuracy: {evaluation['top1_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {evaluation['top5_accuracy'] * 100:.2f}%")

# --- ENSEMBLE PARSER ---

def handle_ensemble_parser(args):
    assert len(args.models) == len(args.modalities) == len(args.alphas), \
        "Mismatch: number of models, modalities, and alphas must be equal"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    datasets = []
    for model_path, modality in zip(args.models, args.modalities):
        
        # Load model
        model = ms_aagcn().to(device)
        model_sd, _, _, _, batch_size = load_model(model_path)
        cleaned_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
        model.load_state_dict(cleaned_sd, strict=False)
        models.append(model)

        # Load dataset (modality-aware)
        ds = NTU_Dataset(root=args.datasets,
                         pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                         pre_transform=NTU_Dataset.__nturgbd_pre_transformer__,
                         modality=modality,
                         benchmark=args.benchmark,
                         part="eval",
                         extended=args.extended)
        datasets.append(ds)

    # Use shared batch_size (assumes same for all models)
    evaluator = EnsembleEvaluator(models=models, datasets=datasets, alphas=args.alphas, batch_size=batch_size)
    evaluation = evaluator.evaluate(display=args.display)

    print(f"Top-1 Accuracy: {evaluation['top1_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {evaluation['top5_accuracy'] * 100:.2f}%")

# ==================== CLI Interface ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-streams Attention Adaptive model for Human's Action Recognition")
    parser.add_argument("--disable-cuda", action="store_true", help="disable CUDA")

    mode_parser = parser.add_subparsers(dest="mode", required=True, help="select mode: process | train | eval | ensemble")

    # --- Process Mode ---
    process_parser = mode_parser.add_parser("process", help="Process Data")

    process_parser.add_argument("--dataset", type=str, default="data/nturgb+d_skeletons/", help="path towards dataset")
    process_parser.add_argument("--extended", action="store_true", help="use NTU RGB+D 120 dataset")
    process_parser.add_argument("--modality", default="joint", choices=["joint", "bone"], help="modality: joint | bone")
    process_parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview", "xsetup"], help="benchmark: xsub | xview | xsetup")
    process_parser.add_argument("--part", default="train", choices=["train", "eval"], help="part: train | eval")
    process_parser.add_argument("--pre_transform", action="store_true", help="authorize pre-transformation")
    process_parser.add_argument("--summary", action="store_true", help="summary of preprocessed dataset")

    process_parser.set_defaults(func=handle_process_parser)

    # --- Train Mode ---
    train_parser = mode_parser.add_parser("train", help="Train Model")

    train_parser.add_argument("--dataset", type=str, default="data/nturgb+d_skeletons/", help="path towards dataset")
    train_parser.add_argument("--extended", action="store_true", help="use NTU RGB+D 120 dataset")
    train_parser.add_argument("--modality", default="joint", choices=["joint", "bone"], help="modality: joint | bone")
    train_parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview", "xsetup"], help="benchmark: xsub | xview | xsetup")
    train_parser.add_argument("--validation", action="store_true", help="enable validation") # TO ADD
    train_parser.add_argument("--pre_transform", action="store_true", help="authorize pre-transformation")
    train_parser.add_argument("--n_epochs", type=int, default=50, help="number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    train_parser.add_argument("--summary", action="store_true", help="summary of model") 

    train_parser.set_defaults(func=handle_train_parser)

    # --- Eval Mode ---
    eval_parser = mode_parser.add_parser("eval", help="Eval Mode")

    eval_parser.add_argument("--dataset", type=str, default="data/nturgb+d_skeletons/", help="Path to dataset")
    eval_parser.add_argument("--extended", action="store_true", help="use NTU RGB+D 120 dataset")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--modality", default="joint", choices=["joint", "bone"], help="Modality: joint | bone")
    eval_parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview", "xsetup"], help="benchmark: xsub | xview | xsetup")
    eval_parser.add_argument("--display", action="store_true", help="Display predictions")

    eval_parser.set_defaults(func=handle_eval_parser)

    # --- Ensemble Mode ---
    ensemble_parser = mode_parser.add_parser("ensemble", help="Evaluate an ensemble of models")

    ensemble_parser.add_argument("--models", nargs='+', required=True, help="Paths to model files")
    ensemble_parser.add_argument("--datasets", type=str, default="data/nturgb+d_skeletons/", help="Paths to corresponding datasets")
    ensemble_parser.add_argument("--extended", action="store_true", help="use NTU RGB+D 120 dataset")
    ensemble_parser.add_argument("--modalities", nargs='+', required=True, help="Modalities per dataset (joint, bone, etc.)")
    ensemble_parser.add_argument("--alphas", nargs='+', type=float, required=True, help="Weight per model")
    ensemble_parser.add_argument("--benchmark", default="xsub", choices=["xsub", "xview", "xsetup"], help="benchmark: xsub | xview | xsetup")
    ensemble_parser.add_argument("--display", action="store_true", help="Show predictions vs ground truth")
    ensemble_parser.set_defaults(func=handle_ensemble_parser)

    # --- CLI Interface ---

    args = parser.parse_args()

    if args.disable_cuda:
        device = torch.device('cpu')
    """
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    """ 

    # print(f"Using {device}", file=sys.stderr)

    args.func(args)
