from src.ai.mcts import EXPLORATION, MCTS
from src.alphazero.evaluator import NetEvaluator
from src.alphazero.network import AlphaZeroNet
from src.connect_four.game import C4Game
from src.connect_four.move import C4Move
from src.connect_four.state import C4State


def build_net(game: C4Game) -> AlphaZeroNet:
    """A freshly initialised network shaped for `game`'s board."""
    board = game.initial().board
    height, width = len(board), len(board[0])

    # One action per column
    return AlphaZeroNet(height=height, width=width, n_actions=width)


class AlphaZero(MCTS[C4State, C4Move]):
    """The MCTS agent, with the network judging every leaf.

    Selection, backup and the final choice are MCTS's own, untouched; only the
    priors and the leaf value change hands.
    """

    net: AlphaZeroNet

    def __init__(
        self,
        game: C4Game,
        net: AlphaZeroNet,
        n_simulations: int,
        exploration: float = EXPLORATION,
        seed: int | None = None,
    ):
        super().__init__(
            game,
            n_simulations=n_simulations,
            evaluate_leaf=NetEvaluator(net),
            exploration=exploration,
            seed=seed,
        )

        self.net = net

    @property
    def description(self) -> str:
        return f"alphazero(simulations={self.n_simulations})"
