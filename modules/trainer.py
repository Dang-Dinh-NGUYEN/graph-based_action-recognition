import gc
import sys
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from modules.utils import *

import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

class Trainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model:nn.Module, optimizer, loss_function, batch_size, rank=0, world_size=1, patience=5, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = device if device is not None else torch.device(f"cuda:{self.rank}")
        self.scaler = GradScaler('cuda')
        # Early stopping variables
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None  # Store the best model

        self.history = {"train_loss": [],
                        "dev_loss": []}
        
    def train(self, train_data : Dataset, val_data : Dataset, num_epoch):
        train_sampler = DistributedSampler(train_data, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_data, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        train_data_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler, 
                                        num_workers=4,          # tune depending on CPU
                                        pin_memory=True,        # faster CPU→GPU transfer
                                        prefetch_factor=2,       # optional, default is fine
                                        persistent_workers=True
                                       )
        
        val_data_loader = DataLoader(val_data, batch_size=self.batch_size, sampler=val_sampler, 
                                        num_workers=4,          # tune depending on CPU
                                        pin_memory=True,        # faster CPU→GPU transfer
                                        prefetch_factor=2,       # optional, default is fine 
                                        persistent_workers=True
                                    )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 40], gamma=0.1)

        for epoch in range(num_epoch):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            self.model.train()
            avg_train_loss = self._run_epoch(train_data_loader, train=True)
            
            self.model.eval()
            avg_val_loss = self._run_epoch(val_data_loader, train=False)

            if self.rank == 0:
                print(f"Epoch {epoch + 1}/{num_epoch}: Training Loss = {avg_train_loss:.4f} Val Loss = {avg_val_loss:.4f}")   

            
            # Early Stopping Check
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
            else:
                self.epochs_without_improvement += 1
            
            """
            # Stop training if patience is exceeded
            if self.epochs_without_improvement >= self.patience:
                break
            """
    
            if self.scheduler:
                self.scheduler.step()

        # Restore the best model before exiting
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0

        data_iter = tqdm(dataloader, unit="batch") if self.rank == 0 else dataloader
        for batch in data_iter:
            self.optimizer.zero_grad()
            # batch = batch.to(self.device)

            inputs = batch.x.to(self.device)
            truths = (batch.y - 1).to(self.device)

            if train:
                with autocast('cuda'):  # Forward pass in mixed precision
                    outputs = self.model(inputs)
                    if torch.isnan(outputs).any():
                        print("Model output contains NaNs!")
                        print(outputs)

                    loss = self.compute_loss(outputs, truths) # + self.model.kl_loss()

                self.scaler.scale(loss).backward()         # Scaled backprop
                self.scaler.step(self.optimizer)           # Scaled optimizer step
                self.scaler.update()                       # Update scale for next step
            else:
                with torch.no_grad():
                    with autocast('cuda'):
                        outputs = self.model(inputs)
                        if torch.isnan(outputs).any():
                            print("Model output contains NaNs!")
                            print(outputs)

                        loss = self.compute_loss(outputs, truths)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.history[f'{"train" if train else "dev"}_loss'].append(avg_loss)

        return avg_loss

    def compute_loss(self, outputs, truths):
        return self.loss_function(outputs, truths)
    
    
