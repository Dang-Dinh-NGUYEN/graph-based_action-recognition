import math
import os
import time
from typing import Mapping
import torch
import random

import data_generator
from data_generator.utils.augmentation import  Shear, TemporalCrop
from modules.loss import LossStrategy
from modules.utils import *
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import Compose
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import torch.distributed as dist


def build_batch_handler(config, *args, **kwargs):
    batch_handler_type = config["type"].lower()
    if batch_handler_type not in data_generator.BATCH_HANDLER_REGISTRY:
        raise ValueError(f"Unknown batch sampler: {batch_handler_type}")
    kwargs = {k: v for k, v in kwargs.items() if k != "type"}
    return data_generator.BATCH_HANDLER_REGISTRY[batch_handler_type](config, *args, **kwargs)

class PreTrainer:
    def __init__(self, model:nn.Module, modality:str, loss_strategy: LossStrategy, optimizer, scheduler,
                batch_config=None, writer=None, autograd=False,
                rank=0, world_size=1, device=None):
        self.model = model
        self.modality=modality
        self.loss_strategy = loss_strategy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = 0
        self.augmentation = Compose([TemporalCrop(), Shear()])

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

        self.history = {'train_loss': [], 'val_loss': []}

        if self.rank == 0:
            self.writer = writer
            self.writer.watch(self.model, log=None, log_freq=1)
        else:
            os.environ["WANDB_MODE"] = "disabled"

    def train(self, train_data, num_epoch, checkpoint_dir=None, checkpoint_interval=5, from_scratch=False):
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

            if checkpoints and not from_scratch:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f"Loading checkpoint: {checkpoint_path}")
                model_sd, optim, hist = load_model(checkpoint_path)
                self.model.load_state_dict(model_sd)
                self.history = hist
                start_epoch = len(hist["train_loss"])
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("Starting from scratch.")
                start_epoch = 0

        self.train_batch_handler = build_batch_handler(config=self.batch_config, dataset_raw=train_data, rank=self.rank, world_size=self.world_size, shuffle=True)
        self.train_data_loader = self.train_batch_handler.build_dataloader()

        for epoch in range(start_epoch, num_epoch):
            self.train_data_loader.batch_sampler.set_epoch(epoch)

            avg_train_loss = self._run_epoch(self.train_data_loader, self.train_batch_handler, train=True, epoch=epoch)

            if self.rank == 0:
                print(f"{epoch + 1}/{num_epoch}:")
                for k, v in avg_train_loss.items():
                    self.writer.log({f"Loss/Train_{k}": v}, step=epoch)
                    print(f"avg_train_{k}: {v:.4f}")
               
                self.writer.log({"LR": self.optimizer.param_groups[0]["lr"]}, step=epoch)

                if checkpoint_dir and ((epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epoch):
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    save_model(checkpoint_path, self.model, self.optimizer, self.history) 

            if self.scheduler is not None:
                self.scheduler.step()

    def _run_epoch(self, dataloader, batch_handler, train: bool, epoch: int):
        total_loss = {}
        data_iter = tqdm(dataloader, unit="batch", dynamic_ncols=True, mininterval=0.1) if self.rank == 0 else dataloader

        for batch in data_iter:
            self.optimizer.zero_grad()
            batch_dict = batch_handler.load_batch(batch, self.device)
            
            frozen_features = batch_dict.get("frozen_features", None)
            raw_inputs = batch_dict.get("inputs", None)['raw_inputs']

            with torch.no_grad():   
            # input = raw_inputs[0].to(self.device)        
                input = self.augmentation(raw_inputs).to(self.device)             
            # input2 = raw_inputs[1].to(self.device)
                input2 = self.augmentation(raw_inputs).to(self.device)     

            if self.modality == "bone":
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone1 = torch.zeros_like(input).to(self.device)
                bone2 = torch.zeros_like(input2).to(self.device)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = input[:, :, :, v1 - 1, :] - input[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = input2[:, :, :, v1 - 1, :] - input2[:, :, :, v2 - 1, :]

                input = bone1
                input2 = bone2
            
            elif self.modality == "motion":
                motion1 = torch.zeros_like(input).to(self.device)
                motion2 = torch.zeros_like(input2).to(self.device)

                motion1[:, :, :-1, :, :] = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = input2[:, :, 1:, :, :] - input2[:, :, :-1, :, :]

                input = motion1
                input2 = motion2
           
            model_inputs = {"raw_input": raw_inputs, "anchor": input, "positive": input2}
            
            if train:
                if self.autograd:
                    with autocast():
                        output = self.model(model_inputs, frozen_features=frozen_features, train=True)
                else: 
                    output = self.model(model_inputs, frozen_features=frozen_features, train=True)

                loss_dict = self.loss_strategy.compute_loss(output, batch_dict["labels"], epoch=epoch, step=self.step)
                dist.all_reduce(loss_dict["loss"], op=dist.ReduceOp.SUM)
                loss_dict["loss"] /= 2

                if self.autograd and self.scaler:
                    self.scaler.scale(loss_dict["loss"]).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict["loss"].backward()
                    self.optimizer.step()
    
            for key, val in loss_dict.items():
                total_loss[key] = total_loss.get(key, 0.0) + val

            self.step += 1

        avg_loss = {f"{key}": total / len(dataloader) for key, total in total_loss.items()}
        self.history['train_loss' if train else 'val_loss'].append(avg_loss)

        return avg_loss
