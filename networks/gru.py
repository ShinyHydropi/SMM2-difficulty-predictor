import networks.data_loader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import random
from networks.training_utils import train_network

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

def pooled(n):
    return (n - 1) // 2 + 1

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class WorldBlock(nn.Module):
    def __init__(self, hidden_size = 64, vertical = False):
        super().__init__()
        permutation = Permute(0, 2, 1, 3) if vertical else Permute(0, 3, 1, 2)
        flat_dim_magnitude = 12 if vertical else 7
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            permutation,
            nn.Flatten(-2, -1)
        )
        self.gru = nn.GRU(input_size=16 * flat_dim_magnitude, hidden_size=hidden_size, batch_first=True)
    def forward(self, x, lengths):
        seq = self.conv(x)
        lengths = pooled(pooled(lengths.long()))
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        return self.gru(packed)[1][-1]

class LevelGRU(nn.Module):
    def __init__(self, gru_hidden_size = 64, fc_hidden_size = 256, scalar_inputs = networks.data_loader.RAW_SCALAR_COUNT):
        super().__init__()
        self.overworld = WorldBlock(hidden_size=gru_hidden_size)
        self.subworld_horizontal = WorldBlock(hidden_size=gru_hidden_size)
        self.subworld_vertical = WorldBlock(hidden_size=gru_hidden_size, vertical=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=gru_hidden_size * 3 + scalar_inputs, out_features=fc_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x, device):
        cols = networks.data_loader.SIZE_COLUMNS
        sizes = x["scalars"]

        has_h = x["subworldHorizontal"].flatten(1).any(1)
        has_v = x["subworldVertical"].flatten(1).any(1)

        overworld = self.overworld(
            x["overworld"].to(device),
            sizes[:, cols["overworldWidth"]],
        )

        subworld_horizontal = torch.zeros_like(overworld)
        if has_h.any():
            subworld_horizontal[has_h] = self.subworld_horizontal(
                x["subworldHorizontal"][has_h].to(device),
                sizes[has_h, cols["subworldWidth"]],
            )

        subworld_vertical = torch.zeros_like(overworld)
        if has_v.any():
            subworld_vertical[has_v] = self.subworld_vertical(
                x["subworldVertical"][has_v].to(device),
                sizes[has_v, cols["subworldHeight"]],
            )

        features = torch.cat((x["scalars"].to(device), overworld, subworld_horizontal, subworld_vertical), dim=1)
        return self.fc_layers(features).reshape(-1)

    @staticmethod
    def name():
        return "LevelGRU"

for lr, epochs in [(1e-3, 100), (1e-4, 200)]:
    fix_random_seed(444)

    level_net = LevelGRU().to(device=device)
    loader = networks.data_loader.loadGru("train")
    valid = networks.data_loader.loadGru("val")

    train_network(level_net, loader, valid, device, epochs = epochs, lr = lr)