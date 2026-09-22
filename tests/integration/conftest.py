from typing import Protocol

import pytest

from src.tic_tac_toe import TTTGame


class TTTFactory(Protocol):
    def __call__(self) -> TTTGame: ...


@pytest.fixture
def ttt() -> TTTFactory:
    def _ttt() -> TTTGame:
        return TTTGame()

    return _ttt
