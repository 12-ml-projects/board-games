from typing import Final, Protocol, Sequence
from uuid import UUID

from src.tournament.match import MatchRecord

BASELINE_ELO: Final[float] = 1500.0


class EloEstimator(Protocol):
    def __call__(
        self, records: Sequence[MatchRecord], *args, **kwargs
    ) -> dict[UUID, float]: ...
