import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from tqdm import tqdm
from collections import defaultdict

class Evaluator:
    def __init__(self, model, device=None, batch_size=32, rank=0, world_size=1):
        self.model =  model 
        self.rank = rank
        self.world_size = world_size
        self.device = device if device else torch.device(f"cuda:{rank}")
        self.batch_size = batch_size

        model.eval()

    def evaluate(self, dataset, topk=(1, 5), display=False):
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                                num_workers=4, pin_memory=True, persistent_workers=True)

        correct_topk = defaultdict(int)
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                x = batch.x.to(self.device)
                y = (batch.y - 1).to(self.device)

                with autocast('cuda'):
                        output = self.model(x)

                total += y.size(0)
                for k in topk:
                    _, pred = output.topk(k, dim=1)
                    correct_topk[k] += (pred == y.unsqueeze(1)).sum().item()

                if display:
                    print(f"Preds: {output.argmax(dim=1)}")
                    print(f"True : {y}")

        results = {
            f"top{k}_accuracy": correct_topk[k] / total for k in topk
        }
        return results
    

class EnsembleEvaluator:
    def __init__(self, models, datasets, alphas, batch_size=32, rank=0, world_size=1):
        self.models = models
        self.datasets = datasets
        self.alphas = alphas
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

    def evaluate(self, topk=(1, 5), display=False):
        dataloaders = [
            DataLoader(
                dataset,
                sampler=DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False),
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
            ) for dataset in self.datasets
        ]
        
        num_batches = len(dataloaders[0])
        dataloaders = [iter(dl) for dl in dataloaders]

        for ds in self.datasets:
            print(len(ds))

        correct_topk = defaultdict(int)
        total = 0

        pbar = tqdm(range(num_batches), desc="Evaluating", disable=(self.rank != 0))

        with torch.no_grad():
            for _ in pbar:
                batches = [next(dl) for dl in dataloaders]
              
                # Assume all batches have the same targets
                y = (batches[0].y - 1).to(self.device)  # assuming class labels are 1-based
                total += y.size(0)

                combined_output = None
               
                with autocast('cuda'):
                    for model, batch, alpha in zip(self.models, batches, self.alphas):
                        x = batch.x.to(self.device)
                        out = model(x)
                        combined_output = out * alpha if combined_output is None else combined_output + out * alpha

                for k in topk:
                    _, pred = combined_output.topk(k, dim=1)
                    correct_topk[k] += (pred == y.unsqueeze(1)).sum().item()

                if display and self.rank == 0:
                    print(f"True: {y.cpu().numpy()}")
                    print(f"Pred: {combined_output.argmax(dim=1).cpu().numpy()}")

        results = {f"top{k}_accuracy": correct_topk[k] / total for k in topk}
        return results
 