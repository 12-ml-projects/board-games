from typing import Protocol

import pytest

from src.ai.random import RandomAI
from src.game.game import Game


class RandomAIFactory(Protocol):
    def __call__(self, game: Game) -> RandomAI: ...


@pytest.fixture
def random_ai() -> RandomAIFactory:
    def _random_ai(game: Game) -> RandomAI:
        return RandomAI(game)

    return _random_ai
