import random
from typing import Final, Sequence

from src.ai.ai import AI
from src.exceptions import NoLegalMovesException
from src.game.game import Game
from src.game.move import Move
from src.game.state import State

INFINITY: Final[float] = float("inf")


SOONER_IS_BETTER: Final[float] = 0.01


class _BudgetExhausted(Exception):
    pass


class MiniMax[StateT: State, MoveT: Move](AI[StateT, MoveT]):
    max_nodes: int
    _rng: random.Random
    _nodes: int
    _truncated: bool

    def __init__(
        self,
        game: Game[StateT, MoveT],
        max_nodes: int,
        seed: int | None = None,
    ):
        super().__init__(game)

        self.max_nodes = max_nodes
        self._rng = random.Random(seed)
        self._nodes = 0
        self._truncated = False

    def choose_move(self, state: StateT) -> MoveT:
        moves = list(self._game.legal_moves(state))

        if not moves:
            raise NoLegalMovesException("No legal moves available")

        # Shuffling prevents equal players from playing the same game
        self._rng.shuffle(moves)

        self._nodes = 0
        best_move = moves[0]
        depth = 1

        while True:
            self._truncated = False

            try:
                best_move = self._search_root(state, moves, depth)
            except _BudgetExhausted:
                break

            if not self._truncated:
                break

            depth += 1

        return best_move

    def _search_root(self, state: StateT, moves: Sequence[MoveT], depth: int) -> MoveT:
        best_move = moves[0]
        alpha = -INFINITY

        for move in moves:
            child = self._game.make_move(state, move)
            value = -self._negamax(child, depth - 1, -INFINITY, -alpha)

            if value > alpha:
                alpha = value
                best_move = move

        return best_move

    def _negamax(self, state: StateT, depth: int, alpha: float, beta: float) -> float:
        if self._nodes >= self.max_nodes:
            raise _BudgetExhausted

        self._nodes += 1

        if self._game.is_final_state(state):
            return self._game.evaluate(state) * (1.0 + depth * SOONER_IS_BETTER)

        if depth == 0:
            self._truncated = True

            return self._game.evaluate(state)

        # TODO: could order moves here for better pruning
        value = -INFINITY

        for move in self._game.legal_moves(state):
            child = self._game.make_move(state, move)
            value = max(value, -self._negamax(child, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    @property
    def description(self) -> str:
        return f"minimax(max_nodes={self.max_nodes})"
