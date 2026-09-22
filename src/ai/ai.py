from abc import ABC, abstractmethod

from src.game import Game, Move, State


class AI(ABC):
    _game: Game

    def __init__(self, game: Game):
        self._game = game

    @abstractmethod
    def choose_move(self, state: State) -> Move:
        pass
