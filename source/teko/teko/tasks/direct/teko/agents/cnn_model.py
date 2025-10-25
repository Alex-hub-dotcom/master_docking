# /workspace/teko/source/teko/teko/tasks/direct/teko/agents/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import Model


class CNNPolicy(Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=False):
        super().__init__(observation_space, action_space, device, clip_actions, clip_log_std)

        # input shape: (batch, channels, height, width)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),   # -> (16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),  # -> (32, H/4, W/4)
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute output size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape).to(device)
            conv_out = self.cnn(sample)
            conv_size = conv_out.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(conv_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def compute(self, observations, **kwargs):
        x = self.cnn(observations)
        actions = self.mlp(x)
        return actions, {}  # empty dict for extra info


class CNNValue(Model):
    def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape).to(device)
            conv_out = self.cnn(sample)
            conv_size = conv_out.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(conv_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, observations, **kwargs):
        x = self.cnn(observations)
        value = self.mlp(x)
        return value, {}
