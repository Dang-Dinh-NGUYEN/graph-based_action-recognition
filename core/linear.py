import math
import os
import time
import torch
import random
import torch.distributed as dist

import modules, data_generator
from modules.loss import LossStrategy
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
from modules.utils import *
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import copy

def _state_dict_to_cpu(sd):
    return {k: v.cpu().clone() for k, v in sd.items()}

def build_batch_handler(config, *args, **kwargs):
    batch_handler_type = config["type"].lower()
    if batch_handler_type not in data_generator.BATCH_HANDLER_REGISTRY:
        raise ValueError(f"Unknown batch sampler: {batch_handler_type}")
    kwargs = {k: v for k, v in kwargs.items() if k != "type"}
    return data_generator.BATCH_HANDLER_REGISTRY[batch_handler_type](config, *args, **kwargs)

def _get_module(m):
    return m.module if hasattr(m, "module") else m

def _ddp_inited():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _extract_targets(labels: torch.Tensor) -> torch.Tensor:
    """
    Make sure labels are class indices on the same device.
    Supports:
      - indices: shape [B] or [B,1]
      - one-hot: shape [B, C] (float/bool) -> argmax
    """
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels#.squeeze(1)
    if labels.dim() == 2 and labels.size(1) > 1:
        # one-hot (float/bool)
        labels = labels.argmax(dim=1)
    return labels.long()

class Linear:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model: nn.Module, modality:str, loss_strategy: LossStrategy, optimizer, scheduler,
                 batch_config=None, writer=None, autograd=False, rank=0, world_size=1, device=None):
        self.model = model
        self.loss_strategy = loss_strategy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.modality = modality
        self.rank = rank
        self.world_size = world_size
        self.device = device if device is not None else torch.device(f"cuda:{self.rank}")
        print(f"current device: {self.device}")

        self.batch_config = batch_config
        self.autograd = autograd
        if self.autograd:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.best_val_acc = -1.0
        self.best_model_state = None
        self.history = {'train_loss': [], 'val_acc': []}

        self.writer = writer if rank == 0 else None
        if self.writer is None:
            os.environ["WANDB_MODE"] = "disabled"
        else:
            # only call watch when writer exists
            try:
                self.writer.watch(self.model, log=None, log_freq=1)
            except Exception:
                pass

    def train(self, train_data, val_data, num_epoch, checkpoint_dir=None, checkpoint_interval=5, from_scratch=False):
        start_epoch = 0
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
                start_epoch = len(hist.get("train_loss", []))
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("Starting from scratch.")

        self.train_batch_handler = build_batch_handler(
            config=self.batch_config, dataset_raw=train_data,
            rank=self.rank, world_size=self.world_size, shuffle=True
        )
        self.train_data_loader = self.train_batch_handler.build_dataloader()

        self.val_batch_handler = build_batch_handler(
            config=self.batch_config, dataset_raw=val_data,
            rank=self.rank, world_size=self.world_size, shuffle=False
        )
        self.val_data_loader = self.val_batch_handler.build_dataloader()

        for epoch in range(start_epoch, num_epoch):
            # set epoch for (Distributed)Samplers
            if hasattr(self.train_data_loader.batch_sampler, "set_epoch"):
                self.train_data_loader.batch_sampler.set_epoch(epoch)
            if hasattr(self.val_data_loader.batch_sampler, "set_epoch"):
                self.val_data_loader.batch_sampler.set_epoch(epoch)

            # ---- Training ----
            avg_train_loss, _ = self._run_epoch(self.train_data_loader, self.train_batch_handler, train=True, epoch=epoch)

            # ---- Conditional evaluation ----
            should_eval = ((epoch + 1) % 1 == 0) or ((epoch + 1) == num_epoch)
            if should_eval:
                avg_val_loss, acc = self._run_epoch(self.val_data_loader, self.val_batch_handler, train=False, epoch=epoch)
            else:
                avg_val_loss, acc = {}, None  # skip evaluation

            # ---- Logging (rank 0 only) ----
            if self.rank == 0:
                print(f"{epoch + 1}/{num_epoch}:")
                for k, v in avg_train_loss.items():
                    if self.writer: self.writer.log({f"Loss/Train_{k}": v}, step=epoch)
                    print(f"avg_train_{k}: {v:.4f}")

                # Only log validation when evaluated
                if should_eval:
                    for k, v in avg_val_loss.items():
                        if self.writer: self.writer.log({f"Loss/Val_{k}": v}, step=epoch)
                        print(f"avg_val_{k}: {v:.4f}")

                    if acc is not None:
                        if self.writer: self.writer.log({"Accuracy/Val": acc["top1_accuracy"]}, step=epoch)
                        print(f"val_acc: {acc['top1_accuracy']*100:.2f}%")
                        

                        # Save best model by accuracy
                        if acc["top1_accuracy"] > self.best_val_acc:
                            self.best_val_acc = acc["top1_accuracy"]
                            self.best_model_state = copy.deepcopy(self.model.state_dict())
                            print("Best model updated!")

                        print(f"best result: {self.best_val_acc * 100:.2f}%")

                # Log LR
                lr = self.optimizer.param_groups[0]["lr"]
                if self.writer: self.writer.log({"LR": lr}, step=epoch)

                # ---- Checkpoint every 5 or final ----
                if checkpoint_dir and ((epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epoch):
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")

                    best_model = copy.deepcopy(self.model)
                    best_model.load_state_dict(self.best_model_state)
                    save_model(checkpoint_path, self.model, self.optimizer, self.history)

            if self.scheduler is not None:
                self.scheduler.step()

    def _run_epoch(self, dataloader, batch_handler, train: bool, epoch: int, topk=(1, 5)):
        """
        DDP-safe metric aggregation, aligned with Evaluator:
        - Computes top-k accuracy (multiclass).
        - Loss aggregated by number of samples.
        """
        model = _get_module(self.model)

        sum_loss = defaultdict(float)
        num_samples = 0
        correct_topk = {k: 0 for k in topk}
        step = 0

        iterator = tqdm(dataloader, unit="batch", dynamic_ncols=True, mininterval=0.1) if self.rank == 0 else dataloader

        for batch in iterator:
            self.optimizer.zero_grad(set_to_none=True)
            batch_dict = batch_handler.load_batch(batch, self.device)
            
            inputs = batch_dict["inputs"]
            inputs["raw_inputs"] = inputs["raw_inputs"].to(self.device, non_blocking=True)
            # labels = _extract_targets(batch_dict["labels"]).to(self.device, non_blocking=True)
            # print(inputs["raw_inputs"].shape)
            labels = batch_dict["labels"].to(self.device, non_blocking=True)
            # labels = _extract_targets(batch_dict["labels"]).to(self.device, non_blocking=True)

            if self.modality == "bone":                
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone = torch.zeros_like(inputs["raw_inputs"]).to(self.device)
                
                for v1, v2 in Bone:
                    bone[:, :, :, v1 - 1, :] = inputs["raw_inputs"][:, :, :, v1 - 1, :] - inputs["raw_inputs"][:, :, :, v2 - 1, :]

                inputs["raw_inputs"] = bone
                
            elif self.modality == "motion":
                    
                motion = torch.zeros_like(inputs["raw_inputs"]).to(self.device)

                motion[:, :, :-1, :, :] = inputs["raw_inputs"][:, :, 1:, :, :] - inputs["raw_inputs"][:, :, :-1, :, :]

                inputs["raw_inputs"] = motion

            if train:
                model.train()
                
                if self.autograd:
                    with autocast():
                        out = model(inputs)
                else:
                    out = model(inputs)

                logits = out["logits"].float()
                loss_dict = self.loss_strategy.compute_loss(out, labels, epoch=epoch, step=step)
                dist.all_reduce(loss_dict["loss"], op=dist.ReduceOp.SUM)
                loss_dict["loss"] /= 2

                if self.autograd and self.scaler:
                    self.scaler.scale(loss_dict["loss"]).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict["loss"].backward()
                    self.optimizer.step()

                bs = labels.size(0)
                for k, v in loss_dict.items():
                    sum_loss[k] += float(v) * bs
                num_samples += bs
                step += 1
            
            else:
                model.eval()
                with torch.no_grad():
                    if self.autograd:
                        with autocast():
                            out = model(inputs)
                    else:
                        out = model(inputs)

                logits = out["logits"].float()
                loss_dict = self.loss_strategy.compute_loss(out, labels, epoch=epoch, step=step)

                # print("pred[:20]:", logits.argmax(dim=1)[:20].cpu().numpy())
                # print("labels[:20]:", labels[:20].cpu().numpy())
                # print("label max:", labels.max().item(), "label min:", labels.min().item())

            # ---- accuracy (multiclass top-k) ----
            for k in topk:
                _, pred = logits.topk(k, dim=-1)
                pred = pred.t()  # shape: (k, batch)
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                correct_topk[k] += correct[:k].reshape(-1).float().sum(0).item()

            # ---- accumulate weighted loss ----
            bs = labels.size(0)
            for k, v in loss_dict.items():
                sum_loss[k] += float(v) * bs
            num_samples += bs
            step += 1

        # ---- DDP reduce ----
        device = self.device
        num_samples_t = torch.tensor(num_samples, device=device, dtype=torch.long)
        if _ddp_inited() and self.world_size > 1:
            torch.distributed.all_reduce(num_samples_t, op=torch.distributed.ReduceOp.SUM)

        denom = max(1, int(num_samples_t.item()))
        avg_loss = {k: (torch.tensor(v, device=device) / denom).item() for k, v in sum_loss.items()}

        # reduce accuracy
        acc_dict = {}
        for k, v in correct_topk.items():
            v_t = torch.tensor(v, device=device, dtype=torch.long)
            if _ddp_inited() and self.world_size > 1:
                torch.distributed.all_reduce(v_t, op=torch.distributed.ReduceOp.SUM)
            acc_dict[f"top{k}_accuracy"] = (v_t.float() / denom).item()

        # update history
        if train:
            self.history["train_loss"].append(avg_loss)
            self.history.setdefault("train_acc", []).append(acc_dict)
        else:
            self.history["val_acc"].append(acc_dict)

        return avg_loss, acc_dict
