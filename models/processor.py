import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch.nn as nn
from tqdm import tqdm

class Processor:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model:nn.Module, optimizer, loss_function, batch_size, patience=5, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Early stopping variables
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None  # Store the best model

        self.history = {"train_loss": [],
                        "dev_loss": []}
        
    def train(self, train_data : Dataset, val_data : Dataset, num_epoch):
        train_data_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        val_data_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        for epoch in range(num_epoch):
            self.model.train()
            avg_train_loss = self._run_epoch(train_data_loader, train=True)

            self.model.eval()
            avg_val_loss = self._run_epoch(val_data_loader, train=False)

            print(f"Epoch {epoch + 1}/{num_epoch}: Training Loss = {avg_train_loss:.4f} Val Loss = {avg_val_loss:.4f}")   

            # Early Stopping Check
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
            else:
                self.epochs_without_improvement += 1

            # Stop training if patience is exceeded
            if self.epochs_without_improvement >= self.patience:
                break

        # Restore the best model before exiting
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0

        with tqdm(dataloader, unit="batch") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                inputs = batch.x.to(self.device)
                truths = batch.y.to(self.device)

                if not train:
                    with torch.no_grad():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                loss = self.compute_loss(outputs, truths)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss/len(dataloader)

        self.history[f"{"train" if train else "dev"}_loss"].append(avg_loss)

        return avg_loss

    def compute_loss(self, outputs, truths):
        return self.loss_function(outputs, truths)
    


