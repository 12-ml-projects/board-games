import random
from math import sqrt
from typing import Final, Mapping, Protocol

from src.ai.ai import AI
from src.exceptions import NoLegalMovesException
from src.game.game import Game
from src.game.move import Move
from src.game.node import Node
from src.game.outcome import score
from src.game.state import State

# Educated guess
EXPLORATION: Final[float] = 1.5


class LeafEvaluator[StateT: State, MoveT: Move](Protocol):
    def __call__(
        self, game: Game[StateT, MoveT], state: StateT
    ) -> tuple[Mapping[MoveT, float], float]: ...


def evaluate_leaf_uniformly[StateT: State, MoveT: Move](
    game: Game[StateT, MoveT], state: StateT
) -> tuple[Mapping[MoveT, float], float]:
    moves = game.legal_moves(state)
    uniform_move_prob = 1.0 / len(moves) if moves else 0.0

    return {move: uniform_move_prob for move in moves}, game.evaluate(state)


class MCTS[StateT: State, MoveT: Move](AI[StateT, MoveT]):
    n_simulations: int
    exploration: float

    def __init__(
        self,
        game: Game[StateT, MoveT],
        n_simulations: int,
        evaluate_leaf: LeafEvaluator[StateT, MoveT] = evaluate_leaf_uniformly,
        exploration: float = EXPLORATION,
        seed: int | None = None,
    ):
        super().__init__(game)

        self.n_simulations = n_simulations
        self.exploration = exploration
        self._evaluate_leaf = evaluate_leaf
        self._rng = random.Random(seed)

    def choose_move(self, state: StateT) -> MoveT:
        if not self._game.legal_moves(state):
            raise NoLegalMovesException("No legal moves available")

        root: Node[StateT, MoveT] = Node(state=state, prior=1.0)

        for _ in range(self.n_simulations):
            self._run_simulation(root)

        moves = list(root.children)
        self._rng.shuffle(moves)

        return max(moves, key=lambda move: root.children[move].visits)

    def _run_simulation(self, root: Node[StateT, MoveT]) -> None:
        path = [root]
        node = root

        while node.expanded:
            node = node.children[self._most_promising(node)]
            path.append(node)

        self._backprop(path, self._judge(node))

    def _most_promising(self, node: Node[StateT, MoveT]) -> MoveT:
        looks = sqrt(node.visits)

        def worth(move: MoveT) -> float:
            child = node.children[move]

            return -child.value + (
                self.exploration * child.prior * looks / (1 + child.visits)
            )

        return max(node.children, key=worth)

    def _judge(self, node: Node[StateT, MoveT]) -> float:
        state = node.state

        if self._game.is_final_state(state):
            return score(self._game.outcome(state), state.player)

        priors, value = self._evaluate_leaf(self._game, state)

        node.children = {
            move: Node(state=self._game.make_move(state, move), prior=prior)
            for move, prior in priors.items()
        }

        return value

    def _backprop(self, path: list[Node[StateT, MoveT]], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.total_value += value
            value = -value

    @property
    def description(self) -> str:
        return f"mcts(simulations={self.n_simulations})"
