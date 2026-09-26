from math import sqrt
from typing import Final

# Normal quantile for a two-sided 95% interval.
Z: Final[float] = 1.959963985


def wilson_interval(score: float, games: int, z: float = Z) -> tuple[float, float]:
    """A confidence interval for a score share that stays inside [0, 1].

    NOTE: draws counted as half a win make this mildly conservative: half a point is
    less erratic than a coin flip, so the true spread is a little narrower.
    """
    if games <= 0:
        raise ValueError("games must be positive")

    spread = z * z / games
    centre = (score + spread / 2) / (1 + spread)
    half_width = (z / (1 + spread)) * sqrt(
        score * (1 - score) / games + spread / (4 * games)
    )

    return max(0.0, centre - half_width), min(1.0, centre + half_width)
