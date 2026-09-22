from typing import Literal, TypeAlias

from src.player import Player

ONGOING: Literal["ongoing"] = "ongoing"
DRAW: Literal["draw"] = "draw"


Outcome: TypeAlias = Player | Literal["draw", "ongoing"]
