from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.game.state import State

Cell: TypeAlias = Literal["x", "o", ""]


Board: TypeAlias = tuple[tuple[Cell, ...], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class TTTState(State):
    board: Board
