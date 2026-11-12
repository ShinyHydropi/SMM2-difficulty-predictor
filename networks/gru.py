# Imports
from networks.data_loader import loadGru

loader = loadGru("train")

for epoch in range(2):
    print(f"--- Epoch {epoch + 1} ---")
    for batchIndex, batch in enumerate(loader):
        grid, scalars, labels = batch["ow"], batch["scalars"], batch["label"]  # Grids are (batch, 12, height, longest width in batch)
        print(f"Batch {batchIndex}: ow {tuple(grid.shape)}, scalars {tuple(scalars.shape)}, labels {tuple(labels.shape)}")