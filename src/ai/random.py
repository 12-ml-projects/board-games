import random

from src.ai.ai import AI
from src.exceptions import NoLegalMovesException
from src.game.game import Game
from src.game.move import Move
from src.game.state import State


class RandomAI[StateT: State, MoveT: Move](AI[StateT, MoveT]):
    _rng: random.Random

    def __init__(self, game: Game[StateT, MoveT], seed: int | None = None):
        super().__init__(game)

        self._rng = random.Random(seed)

    def choose_move(self, state: StateT) -> MoveT:
        legal_moves = self._game.legal_moves(state)

        if self._game.is_final_state(state):
            raise NoLegalMovesException("No legal moves available")

        return self._rng.choice(legal_moves)
