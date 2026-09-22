from abc import ABC, abstractmethod

from src.game import Game, Move, State


class AI[StateT: State, MoveT: Move](ABC):
    _game: Game[StateT, MoveT]

    def __init__(self, game: Game[StateT, MoveT]):
        self._game = game

    @abstractmethod
    def choose_move(self, state: StateT) -> MoveT:
        pass
