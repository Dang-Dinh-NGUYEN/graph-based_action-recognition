import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from collections import defaultdict

from core.trainer import build_batch_handler
from sklearn.metrics import classification_report

class Evaluator:
    def __init__(self, models, datasets, modalities, alphas, batch_config=None, rank=0, world_size=1, metric="multiclass"):
        """
        metric: "multiclass" or "multilabel"
        """
        self.models = models
        self.datasets = datasets
        self.alphas = alphas
        self.metric = metric
        self.modalities = modalities
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        
        self.batch_config = [batch_config for _ in datasets]
        
    def evaluate(self, topk=(1, 5), display=False, debug=False):
        batch_handlers = [
            build_batch_handler(config=cfg, dataset_raw=ds, rank=self.rank, world_size=self.world_size, shuffle=False)
            for cfg, ds in zip(self.batch_config, self.datasets)
        ]

        dataloaders = [bh.build_dataloader() for bh in batch_handlers]
        num_batches = len(dataloaders[0])
        dataloaders = [iter(dl) for dl in dataloaders]

        if self.metric == "multiclass":
            correct_topk = defaultdict(int)
        else:
            correct_topk = {"accuracy": 0.0}
        total = 0

        pbar = tqdm(range(num_batches), desc="Evaluating", disable=(self.rank != 0))

        with torch.no_grad():
            for step in pbar:
                combined_output = None
                all_labels = []

                for idx, (model, modality, dataloader, batch_handler, alpha) in enumerate(zip(self.models, self.modalities, dataloaders, batch_handlers, self.alphas)):
                    raw_batch = next(dataloader)
                    samples = batch_handler.load_batch(raw_batch, self.device)
                    inputs = samples['inputs']["raw_inputs"].to(self.device, non_blocking=True)
                    
                    if modality == "bone":
                        Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
                            
                        bone = torch.zeros_like(inputs)
                        
                        for v1, v2 in Bone:
                            bone[:, :, :, v1 - 1, :] = inputs[:, :, :, v1 - 1, :] - inputs[:, :, :, v2 - 1, :]

                        inputs = bone
                    
                    elif modality == "motion":
                        motion = torch.zeros_like(inputs)

                        motion[:, :, :-1, :, :] = inputs[:, :, 1:, :, :] - inputs[:, :, :-1, :, :]

                        inputs = motion

                    labels = samples["labels"].to(self.device, non_blocking=True)
                    
                    all_labels.append(labels)
                    
                    # with autocast():
                    out = model(inputs)

                    logits = out["logits"].float()  # ensure float for BCE
                    
                    weighted_logits = logits * alpha

                    if combined_output is None:
                        combined_output = weighted_logits
                    else:
                        combined_output += weighted_logits

                    if debug and self.rank == 0:
                        print(f"\nModel {idx} logits shape: {logits.shape}, alpha: {alpha}")
                        print(f"Logits (first 3): {logits[:3].cpu().numpy()}")

                # Ensure labels are consistent
                for i in range(1, len(all_labels)):
                    if not torch.equal(all_labels[i], all_labels[0]):
                        raise ValueError(f"Label mismatch between model 0 and model {i} at step {step}")
                    
                labels = all_labels[0]
                total += labels.size(0)
                
                if self.metric == "multiclass":
                    for k in topk:
                        val, pred = torch.softmax(combined_output, dim=1).topk(k, dim=1)
                        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
                        correct_topk[k] += correct.any(dim=1).sum().item()
                       
                elif self.metric == "multilabel":
                    # BCE: logits -> sigmoid -> threshold 0.5
                    probs = torch.sigmoid(combined_output)
                    preds = (probs >= 0.5).long()

                    if step == 0:
                        all_preds = preds.cpu()
                        all_targets = labels.cpu()
                    else:
                        all_preds = torch.cat([all_preds, preds.cpu()], dim=0)
                        all_targets = torch.cat([all_targets, labels.cpu()], dim=0)

                if display and self.rank == 0:
                    print(f"Step {step} Pred: {preds.cpu().numpy() if self.metric=='binary' else combined_output.argmax(dim=-1).cpu().numpy()}")
                    print(f"         True: {labels.squeeze().cpu().numpy()}")

        if self.metric == "multiclass":
            results = {f"top{k}_accuracy": correct_topk[k] / total for k in topk}
            print(*[f"top{k}_accuracy: {correct_topk[k] / total * 100:.2f}%" for k in topk], sep="\n")
        else:
            report = classification_report(
                all_targets.numpy(),
                all_preds.numpy(),
                target_names=[f"Class {i}" for i in range(all_targets.shape[1])],
                zero_division=0,
            )
            print(report)

        return results
