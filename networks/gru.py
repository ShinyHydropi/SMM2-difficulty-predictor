import networks.data_loader
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

dl = networks.data_loader.loadGru("train")

for epoch in range(2):
    print(f"--- Epoch {epoch+1} ---")
    for batch_idx, (features, labels) in enumerate(dl):
        # 'features' shape: [16, 5], 'labels' shape: [16]
        print(f"Batch {batch_idx}: Features shape {features.shape}, Labels shape {labels.shape}")