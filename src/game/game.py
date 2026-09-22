from abc import ABC, abstractmethod
from typing import Sequence

from src.game.move import Move
from src.game.outcome import ONGOING, Outcome, score
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

    @abstractmethod
    def material(self, state: StateT) -> float:
        raise NotImplementedError

    @abstractmethod
    def evaluate_material(self, material_advantage: float) -> float:
        raise NotImplementedError

    def is_legal_move(self, state: StateT, move: MoveT) -> bool:
        return move in self.legal_moves(state)

    def is_final_state(self, state: StateT) -> bool:
        return self.outcome(state) != ONGOING

    def evaluate(self, state: StateT) -> float:
        outcome = self.outcome(state)

        if outcome != ONGOING:
            return score(outcome, state.player)

        return self.evaluate_material(self.material(state))
