import networks.data_loader
import torch
import torch.nn as nn
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

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class WorldBlock(nn.Module):
    def __init__(self, hidden_size = 32, vertical = False):
        super().__init__()
        permutation = Permute(0, 2, 1, 3) if vertical else Permute(0, 3, 1, 2)
        flat_dim_magnitude = 12 if vertical else 7
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            permutation,
            nn.Flatten(-2, -1),
            nn.GRU(input_size=16 * flat_dim_magnitude, hidden_size=hidden_size, batch_first=True)
        )
    def forward(self, x):
        return self.net(x)

class LevelGRU(nn.Module):
    def __init__(self, gru_hidden_size = 32, fc_hidden_size = 256, scalar_inputs = 152):
        super().__init__()
        self.overworld = WorldBlock(hidden_size=gru_hidden_size)
        self.subworld_hor = WorldBlock(hidden_size=gru_hidden_size)
        self.subworld_vert = WorldBlock(hidden_size=gru_hidden_size, vertical=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=gru_hidden_size * 3 + scalar_inputs, out_features=fc_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x, device):
        ow = self.overworld(x["ow"].to(device))[0][:, -1, :]
        sw_hor = torch.zeros_like(ow)
        has_sw_hor = x["lengths"]["sw_hor"] != 1
        sw_hor[has_sw_hor, :] = self.subworld_hor(x["sw_hor"][has_sw_hor, :, :, :].to(device))[0][:, -1, :]
        sw_vert = torch.zeros_like(ow)
        has_sw_vert = x["lengths"]["sw_vert"] != 1
        sw_vert[has_sw_vert, :] = self.subworld_vert(x["sw_vert"][has_sw_vert, :, :, :].to(device))[0][:, -1, :]
        scalars = torch.cat((x["scalars"].to(device), ow, sw_hor, sw_vert), dim=1)
        return self.fc_layers(scalars).reshape(-1)

level_net = LevelGRU(gru_hidden_size=32, fc_hidden_size=256, scalar_inputs = 149).to(device=device)

dl = networks.data_loader.loadGru("train")

for epoch in range(2):
    print(f"--- Epoch {epoch+1} ---")
    for batch_idx, batch in enumerate(dl):
        print(f"Batch {batch_idx}: {level_net(batch, device=device)}")
        print(batch["label"].to(device=device))
        quit()