import networks.data_loader
import torch

dl = networks.data_loader.loadGru("train")

for epoch in range(2):
    print(f"--- Epoch {epoch+1} ---")
    for batch_idx, output in enumerate(dl):
        print(f"Batch {batch_idx}: {output["lengths"]}")