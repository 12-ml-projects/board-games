from typing import Protocol

import pytest

from src.ai.random import RandomAI
from src.game.game import Game
from src.game.move import Move
from src.game.state import State


class RandomAIFactory(Protocol):
    def __call__[StateT: State, MoveT: Move](
        self, game: Game[StateT, MoveT]
    ) -> RandomAI[StateT, MoveT]: ...


@pytest.fixture
def random_ai() -> RandomAIFactory:
    def _random_ai[StateT: State, MoveT: Move](
        game: Game[StateT, MoveT],
    ) -> RandomAI[StateT, MoveT]:
        return RandomAI(game)

    return _random_ai
