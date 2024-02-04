from torch import nn
import torch.nn.functional as F
N_ACTIONS = 4


class NN2048(nn.Module):
    def __init__(self, grid_sz=4, layer_sz=128):
        super().__init__()
        self.layer1 = nn.Linear(int(grid_sz**2), layer_sz)
        self.layer2 = nn.Linear(layer_sz, layer_sz)
        self.layer3 = nn.Linear(layer_sz, N_ACTIONS)

    def forward(self, x):
        x = x.flatten()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


if __name__ == '__main__':
    pass
