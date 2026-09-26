from typing import Final, Literal, TypeAlias

from src.player import Player

ONGOING: Literal["ongoing"] = "ongoing"
DRAW: Literal["draw"] = "draw"


Outcome: TypeAlias = Player | Literal["draw", "ongoing"]


WIN: Final[float] = 1.0
LOSS: Final[float] = -1.0
DRAWN: Final[float] = 0.0


# Heuristics stay strictly inside the scores final states own, so that a
# promising board can never be mistaken for a won one.
HEURISTIC_CEILING: Final[float] = 0.9


# An agent may disturb its move values by up to this much and still be certain
# of spotting a win or avoiding a loss.
SAFE_MARGIN: Final[float] = WIN - HEURISTIC_CEILING


def score(outcome: Outcome, player: Player) -> float:
    """Value of `outcome` from `player`'s perspective."""
    if outcome == ONGOING or outcome == DRAW:
        return DRAWN

    return WIN if outcome == player else LOSS
