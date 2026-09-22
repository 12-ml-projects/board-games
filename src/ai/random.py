import random

from src.ai.ai import AI
from src.ai.exceptions import NoLegalMovesException
from src.game.move import Move
from src.game.state import State


class RandomAI[StateT: State, MoveT: Move](AI[StateT, MoveT]):
    def choose_move(self, state: StateT) -> MoveT:
        legal_moves = self._game.legal_moves(state)

        if len(legal_moves) == 0:
            raise NoLegalMovesException("No legal moves available")

        return random.choice(legal_moves)
