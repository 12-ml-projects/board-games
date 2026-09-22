from abc import ABC, abstractmethod
from typing import Sequence

from src.game.move import Move
from src.game.outcome import Outcome
from src.game.state import State


class Game[StateT: State, MoveT: Move](ABC):
    @abstractmethod
    def initial(self) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def legal_moves(self, state: StateT) -> Sequence[MoveT]:
        raise NotImplementedError

    @abstractmethod
    def make_move(self, state: StateT, move: MoveT) -> StateT:
        raise NotImplementedError

    @abstractmethod
    def outcome(self, state: StateT) -> Outcome:
        raise NotImplementedError
