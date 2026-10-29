from typing import Mapping

import torch

from src.alphazero.encoding import encode
from src.alphazero.network import AlphaZeroNet
from src.connect_four.move import C4Move
from src.connect_four.state import C4State
from src.game.game import Game


class NetEvaluator:
    """A LeafEvaluator that asks the network instead of the heuristic.

    The policy head scores every column; full ones are dropped before the
    softmax, so the priors share out all of their mass among the legal moves.
    """

    _net: AlphaZeroNet

    def __init__(self, net: AlphaZeroNet):
        # Batch norm must use its running statistics: a batch of one board has
        # no meaningful statistics of its own.
        self._net = net.eval()

    def __call__(
        self, game: Game[C4State, C4Move], state: C4State
    ) -> tuple[Mapping[C4Move, float], float]:
        moves = game.legal_moves(state)

        with torch.inference_mode():
            logits, value = self._net(encode(state).unsqueeze(0))

        columns = torch.tensor([move.column for move in moves])
        priors = torch.softmax(logits[0, columns], dim=0)

        return dict(zip(moves, priors.tolist())), value.item()
