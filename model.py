import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=450, fc2_units=350, fc3_units=350):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.nonlin = F.relu  # leaky_relu

        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        h1 = self.nonlin(self.bn1(self.fc1(x)))
        h2 = self.nonlin(self.bn2(self.fc2(h1)))
        h4 = self.fc4(h2)
        return F.tanh(h4)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=128):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.nonlin = F.relu  # leaky_relu
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs, act):
        h1 = self.nonlin(self.bn1(self.fc1(obs)))
        h1 = torch.cat((h1, act), dim=1)
        h2 = self.nonlin(self.bn2(self.fc2(h1)))
        h4 = self.fc4(h2)
        return h4
