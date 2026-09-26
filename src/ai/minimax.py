import random
from typing import Final, Sequence

from src.ai.ai import AI
from src.exceptions import NoLegalMovesException
from src.game.game import Game
from src.game.move import Move
from src.game.outcome import SAFE_MARGIN
from src.game.state import State

INFINITY: Final[float] = float("inf")


SOONER_IS_BETTER: Final[float] = 0.01


# No disturbance at all: the search's own choice is always played.
EXACT: Final[float] = 0.0


class _BudgetExhausted(Exception):
    pass


class MiniMax[StateT: State, MoveT: Move](AI[StateT, MoveT]):
    max_nodes: int
    noise: float
    _rng: random.Random
    _nodes: int
    _truncated: bool

    def __init__(
        self,
        game: Game[StateT, MoveT],
        max_nodes: int,
        noise: float = EXACT,
        seed: int | None = None,
    ):
        super().__init__(game)

        if not EXACT <= noise < SAFE_MARGIN:
            raise ValueError(
                f"noise must be at least {EXACT} and below {SAFE_MARGIN}, "
                "or a win could be passed over for a merely promising board"
            )

        self.max_nodes = max_nodes
        self.noise = noise
        self._rng = random.Random(seed)
        self._nodes = 0
        self._truncated = False

    def choose_move(self, state: StateT) -> MoveT:
        moves = list(self._game.legal_moves(state))

        if not moves:
            raise NoLegalMovesException("No legal moves available")

        # Shuffling settles exact ties, which noise alone would leave to the
        # order moves happen to be generated in.
        self._rng.shuffle(moves)

        self._nodes = 0
        scored = [(move, -INFINITY) for move in moves]
        depth = 1

        while True:
            self._truncated = False

            try:
                scored = self._search_root(state, moves, depth)
            except _BudgetExhausted:
                break

            if not self._truncated:
                break

            depth += 1

        return self._pick(scored)

    def _pick(self, scored: Sequence[tuple[MoveT, float]]) -> MoveT:
        """Take the best move, give or take `noise`."""
        if self.noise <= EXACT:
            return max(scored, key=lambda pair: pair[1])[0]

        return max(
            scored, key=lambda pair: pair[1] - self._rng.uniform(EXACT, self.noise)
        )[0]

    def _search_root(
        self, state: StateT, moves: Sequence[MoveT], depth: int
    ) -> list[tuple[MoveT, float]]:
        """Value every root move exactly, on a full window.

        Narrowing the window would let alpha-beta cut a losing move short and
        report a bound instead of its value, which `_pick` would then read as
        a close second.
        """
        return [
            (
                move,
                -self._negamax(
                    self._game.make_move(state, move), depth - 1, -INFINITY, INFINITY
                ),
            )
            for move in moves
        ]

    def _negamax(self, state: StateT, depth: int, alpha: float, beta: float) -> float:
        if self._nodes >= self.max_nodes:
            raise _BudgetExhausted

        self._nodes += 1

        if self._game.is_final_state(state):
            return self._game.evaluate(state) * (1.0 + depth * SOONER_IS_BETTER)

        if depth == 0:
            self._truncated = True

            return self._game.evaluate(state)

        value = -INFINITY
        moves = sorted(self._game.legal_moves(state), key=self._game.move_priority)

        for move in moves:
            child = self._game.make_move(state, move)
            value = max(value, -self._negamax(child, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    @property
    def description(self) -> str:
        return f"minimax(max_nodes={self.max_nodes}, noise={self.noise})"
