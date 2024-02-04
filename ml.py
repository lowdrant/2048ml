#!/usr/bin/env python3
"""
working from pytorch's DQN tutorial:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
from collections import deque, namedtuple

import torch.nn.functional as F
from matplotlib import get_backend as mplgb
from torch import nn, optim
from numpy import exp
from grid import Grid

is_ipython = 'inline' in mplgb()
if is_ipython:
    from IPython import display

N_ACTIONS = 4  # number of actions in 2048
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


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


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainor:
    def __init__(self, grid, nn, eps_start, eps_end, eps_decay):
        self.grid = grid
        self.nn = nn
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            exp(-1. * steps_done / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device,
                                dtype=torch.long)


def plot_durations(show_result=False):
    fig = figure(1)
    ax = fig.axes[0]
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        ax.set_title('Result')
    else:
        fig.clf()
        ax.set_title('Training...')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration')
    ax.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax.plot(means.numpy())

    pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


policy_net = NN2048()
target_net = NN2048()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
