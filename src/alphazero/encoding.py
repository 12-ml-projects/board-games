from typing import Final

import torch

from src.connect_four.state import C4State, Cell
from src.player import Player

# The stones of the side to move, its opponent's, and which colour it plays
N_PLANES: Final[int] = 3


_SYMBOLS: Final[dict[Player, Cell]] = {"A": "x", "B": "o"}


def encode(state: C4State) -> torch.Tensor:
    """The board as (N_PLANES, height, width), seen by the side to move.

    The network only ever sees "mine" and "theirs", never "x" and "o", so one
    set of weights serves both colours. Colour still matters (who moved first
    decides which rows a threat is worth holding), and the stone count that
    implies it is global, out of reach of a 3x3 convolution. So it gets its
    own plane: all ones when A is to move.
    """
    mine = _SYMBOLS[state.player]
    first = state.player == "A"

    return torch.tensor(
        [
            [[cell == mine for cell in row] for row in state.board],
            [[cell not in ("", mine) for cell in row] for row in state.board],
            [[first for _ in row] for row in state.board],
        ],
        dtype=torch.float32,
    )
