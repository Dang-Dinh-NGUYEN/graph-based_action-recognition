import math
import os
import time
import torch
import random

import modules, data_generator
from modules.loss import LossStrategy
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
from modules.utils import *
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Sampler
from collections import OrderedDict, defaultdict
import torch.distributed as dist

def build_batch_handler(config, *args, **kwargs):
    batch_handler_type = config["type"].lower()
    
    if batch_handler_type not in data_generator.BATCH_HANDLER_REGISTRY:
        raise ValueError(f"Unknown batch sampler: {batch_handler_type}")

    # Remove None values from kwargs
    kwargs = {k: v for k, v in kwargs.items() if k != "type"}

    return data_generator.BATCH_HANDLER_REGISTRY[batch_handler_type](config, *args, **kwargs)

# --- Balanced Batch Sampler ---
class DistributedClassBalancedSampler(Sampler):
    def __init__(self, labels, batch_size, num_replicas=None, rank=None, shuffle=True):
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_samples = len(labels)

        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[int(label)].append(idx)

        self.num_batches = math.ceil(len(labels) / batch_size / num_replicas)

    def __iter__(self):
        all_indices = []

        for label, indices in self.label_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            all_indices.extend(indices)

        if self.shuffle:
            random.shuffle(all_indices)

        # Split indices across replicas
        total_size = self.num_batches * self.batch_size * self.num_replicas
        all_indices += all_indices[:(total_size - len(all_indices))]  # pad if needed

        # Subsample for this rank
        indices = all_indices[self.rank::self.num_replicas]
        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return self.num_batches

# --- Trainer ---

class Trainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model:nn.Module, modality:str, loss_strategy: LossStrategy, optimizer, scheduler,
                batch_config=None,
                writer=None, # patience=5,
                autograd=False,
                rank=0, world_size=1, device=None
                ):
        
        self.model = model
        self.modality = modality
        self.loss_strategy = loss_strategy
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.rank = rank
        self.world_size = world_size
        self.device = device if device is not None else torch.device(f"cuda:{self.rank}")

        self.batch_config = batch_config
        self.use_cached_features = batch_config.get("use_cached_features", False)
        self.autograd = autograd
        if self.autograd:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Early stopping variables
        """
        self.patience = patience
        self.epochs_without_improvement = 0
        """
        # self.best_val_loss = float('inf')
        self.best_model_state = None  # Store the best model

        self.history = {'train_loss': [], 'val_loss': []}

        if self.rank == 0:
            self.writer = writer
            self.writer.watch(self.model, log=None, log_freq=1)
        else:
            os.environ["WANDB_MODE"] = "disabled"

    def train(self, train_data, val_data, num_epoch, checkpoint_dir=None, checkpoint_interval=5, from_scratch=False):
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
            if checkpoints and not from_scratch:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f"Loading checkpoint: {checkpoint_path}")
                
                model_sd, optim, hist = load_model(checkpoint_path)
                # cleaned_model_sd = OrderedDict((k.replace("module.", ""), v) for k, v in model_sd.items())
                self.model.load_state_dict(model_sd)
                self.history = hist
                # self.optimizer.load_state_dict(optim)

                start_epoch = len(hist["train_loss"])  # Start from the next epoch
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("Starting from scratch.")
                start_epoch = 0

        self.train_batch_handler = build_batch_handler(config=self.batch_config, dataset_raw=train_data, rank=self.rank, world_size=self.world_size, shuffle=True)
        # self.val_batch_handler = build_batch_handler(config=self.batch_config, dataset_raw=val_data, rank=self.rank, world_size=self.world_size, shuffle=False)
        # print("Batch handlers built")

        self.train_data_loader = self.train_batch_handler.build_dataloader()
        # self.val_data_loader = self.val_batch_handler.build_dataloader()
        # print("Data loader built")

        for epoch in range(start_epoch, num_epoch):
            self.train_data_loader.batch_sampler.set_epoch(epoch)
            # self.val_data_loader.batch_sampler.set_epoch(epoch)

            avg_train_loss = self._run_epoch(self.train_data_loader, self.train_batch_handler, train=True, epoch=epoch)
            # avg_val_loss = self._run_epoch(self.val_data_loader, self.val_batch_handler, train=False, epoch=epoch)

            if self.rank == 0:
                print(f"{epoch + 1}/{num_epoch}:")

                for k, v in avg_train_loss.items():
                    self.writer.log({f"Loss/Train_{k}": v}, step=epoch)
                    print(f"avg_train_{k}: {v:.4f}")

                # for k, v in avg_val_loss.items():
                #     self.writer.log({f"Loss/Val_{k}": v}, step=epoch)
                #     print(f"avg_val_{k}: {v:.4f}")

                lr = self.optimizer.param_groups[0]["lr"]
                self.writer.log({"LR": lr}, step=epoch)
                
                if checkpoint_dir:
                    if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epoch:
                        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                        save_model(checkpoint_path, self.model, self.optimizer, self.history)

            """
            # Early Stopping Check
            if avg_val_loss['loss'] < self.best_val_loss:
                self.best_val_loss = avg_val_loss['loss']
                # self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
            
            
            else:
                self.epochs_without_improvement += 1
            
            # Stop training if patience is exceeded
            if self.epochs_without_improvement >= self.patience:
                break
            """
    
            if self.scheduler is not None:
                self.scheduler.step()    

        # Restore the best model before exiting
        """
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history
        """

    def _run_epoch(self, dataloader, batch_handler, train: bool, epoch: int):
        total_loss = {}
        step = 0

        data_iter = tqdm(dataloader, unit="batch", dynamic_ncols=True, mininterval=0.1) if self.rank == 0 else dataloader

        for batch in data_iter:
            self.optimizer.zero_grad()
            batch_dict = batch_handler.load_batch(batch, self.device)

            # Use frozen features from batch if present, else None
            frozen_features = batch_dict.get("frozen_features", None)
            inputs = batch_dict.get("inputs", None)

            if self.modality == "bone":
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                        (22, 23), (23, 8), (24, 25), (25, 12)]

                bone = torch.zeros_like(inputs["raw_inputs"])
                for v1, v2 in Bone:
                    bone[:, :, :, v1 - 1, :] = inputs["raw_inputs"][:, :, :, v1 - 1, :] - inputs["raw_inputs"][:, :, :, v2 - 1, :]

                inputs["raw_inputs"] = bone

            elif self.modality == "motion":
                motion = torch.zeros_like(inputs["raw_inputs"])

                motion[:, :, :-1, :, :] = inputs["raw_inputs"][:, :, 1:, :, :] - inputs["raw_inputs"][:, :, :-1, :, :]

                inputs["raw_inputs"] = motion

            if train:
                if self.autograd:
                    with autocast():  # Forward pass in mixed precision
                        outputs = self.model(inputs, frozen_features=frozen_features, train=True)
                else:
                    outputs = self.model(inputs, frozen_features=frozen_features, train=True)    

                loss = self.loss_strategy.compute_loss(outputs, batch_dict["labels"], epoch=epoch, step=step)
                dist.all_reduce(loss["loss"], op=dist.ReduceOp.SUM)
                loss["loss"] /= 2

                if self.autograd and self.scaler:
                    self.scaler.scale(loss['loss']).backward()         # Scaled backprop
                    self.scaler.step(self.optimizer)           # Scaled optimizer step
                    self.scaler.update()                       # Update scale for next step
                else:
                    loss["loss"].backward()
                    self.optimizer.step()

            else:
                self.model.eval()
                with torch.no_grad():
                    if self.autograd:
                        with autocast():
                            outputs = self.model(inputs, frozen_features=frozen_features, train=False)
                    else:
                        outputs = self.model(inputs, frozen_features=frozen_features, train=False)

                loss = self.loss_strategy.compute_loss(outputs, batch_dict["labels"], epoch=epoch, step=step)

            for key, val in loss.items():
                total_loss[key] = total_loss.get(key, 0.0) + val

            step += 1

        # Compute average loss for each type
        avg_loss = {f"{key}": total / len(dataloader) for key, total in total_loss.items()}

        phase = 'train' if train else 'val'
        self.history[f"{phase}_loss"].append(avg_loss)

        return avg_loss
