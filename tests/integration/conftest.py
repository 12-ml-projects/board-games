from typing import Protocol

import pytest

from src.connect_four.game import C4Game
from src.tic_tac_toe import TTTGame

DEFAULT_C4_WIDTH = 7
DEFAULT_C4_HEIGHT = 6


class TTTFactory(Protocol):
    def __call__(self) -> TTTGame: ...


class C4Factory(Protocol):
    def __call__(self, width: int = ..., height: int = ...) -> C4Game: ...


@pytest.fixture
def ttt() -> TTTFactory:
    def _ttt() -> TTTGame:
        return TTTGame()

    return _ttt


@pytest.fixture
def c4() -> C4Factory:
    def _c4(width: int = DEFAULT_C4_WIDTH, height: int = DEFAULT_C4_HEIGHT) -> C4Game:
        return C4Game(width=width, height=height)

    return _c4
