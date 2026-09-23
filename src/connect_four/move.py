from dataclasses import dataclass

from src.game.move import Move


@dataclass(frozen=True, slots=True, kw_only=True)
class C4Move(Move):
    column: int
