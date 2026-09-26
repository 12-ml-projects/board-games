from typing import Callable, Mapping

from src.connect_four.game import C4Game
from src.game.game import Game
from src.tic_tac_toe import TTTGame

GAMES: Mapping[str, Callable[[], Game]] = {
    "tic_tac_toe": TTTGame,
    "connect_four": lambda: C4Game(width=7, height=6),
}
