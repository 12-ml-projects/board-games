from typing import Final, Literal, TypeAlias

from src.player import Player

ONGOING: Literal["ongoing"] = "ongoing"
DRAW: Literal["draw"] = "draw"


Outcome: TypeAlias = Player | Literal["draw", "ongoing"]


WIN: Final[float] = 1.0
LOSS: Final[float] = -1.0
DRAWN: Final[float] = 0.0


def score(outcome: Outcome, player: Player) -> float:
    """Value of `outcome` from `player`'s perspective."""
    if outcome == ONGOING or outcome == DRAW:
        return DRAWN

    return WIN if outcome == player else LOSS
