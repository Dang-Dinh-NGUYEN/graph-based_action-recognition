import argparse
from numpy import argmax
import torch
from torch_geometric_temporal import AAGCN
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.functional as nn
from data_generator.ntu_data import RAW_DIR, EDGE_INDEX, NTU_Dataset
from models.ms_aagcn import ms_aagcn

edges = [
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
]

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() - 1  # shape: [2, num_edges]


data_set = NTU_Dataset(root=RAW_DIR, 
                       pre_transform=NTU_Dataset.__nturgbd_pre_transformer__,
                       pre_filter=NTU_Dataset.__nturgbd_pre_filter__,
                       modality="joint",
                       benchmark="xsub",
                       part="train")

train_set = data_set[:100]
val_set = data_set[100:]

train_data = DataLoader(train_set, batch_size=10)
batch = next(iter(train_data))
print(batch)

print(EDGE_INDEX)
print()
print(edge_index)
aagcn_model = ms_aagcn()
# print(aagcn_model)
optimizer = torch.optim.Adam(aagcn_model.parameters(), lr=0.01)

aagcn_model.train()


print("Training...")
for epoch in tqdm(range(1)):
    loss = 0
    step = 0
    for batch in train_data:
        label = batch.y
        batch = batch.x
        y_hat = aagcn_model(batch)
       
        pred = y_hat.argmax(dim=1)      # [B]
        print(pred)  # tensor([class_1, class_2, ..., class_B])
        print(label)
        print()


# ==================== CLI Interface ====================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human's Action Recognition")

