from typing import Final

import torch
from torch import nn

from src.alphazero.encoding import N_PLANES

CHANNELS: Final[int] = 64


N_BLOCKS: Final[int] = 5


# Width of the hidden layer between the value head's convolution and its output
VALUE_HIDDEN: Final[int] = 64


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return torch.relu(out + x)


class AlphaZeroNet(nn.Module):
    """A residual tower with a policy head and a value head.

    Given boards of shape (batch, N_PLANES, height, width), returns one logit
    per action, (batch, n_actions), and the value for the side to move,
    (batch,), squashed into [-1, 1].
    """

    height: int
    width: int
    n_actions: int

    def __init__(
        self,
        height: int,
        width: int,
        n_actions: int,
        channels: int = CHANNELS,
        n_blocks: int = N_BLOCKS,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.n_actions = n_actions
        cells = height * width

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(n_blocks)))

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * cells, n_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cells, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1),
            nn.Tanh(),
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(self.stem(boards))

        return self.policy_head(features), self.value_head(features).squeeze(-1)
