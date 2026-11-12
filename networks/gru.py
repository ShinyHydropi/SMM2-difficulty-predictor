import networks.data_loader
import torch
import random

if torch.cuda.is_available():
    gpu_index = torch.randint(0, torch.cuda.device_count(), (1,)).item()
    device = torch.device('cuda:{}'.format(gpu_index))
    print("PyTorch is using GPU {}!".format(device))
else:
    print("PyTorch is NOT using GPU!")
    quit()


def fix_random_seed(seed_no=0):
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed(seed_no)
    random.seed(seed_no)



dl = networks.data_loader.loadGru("train")

# for epoch in range(2):
#     print(f"--- Epoch {epoch+1} ---")
#     for batch_idx, output in enumerate(dl):
#         print(f"Batch {batch_idx}: {output["lengths"]}")