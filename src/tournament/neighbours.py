from typing import Sequence
from uuid import UUID

from src.tournament.agent_info import AgentInfo
from src.tournament.match_maker import MatchMaker


def neighbours(band: int = 2) -> MatchMaker:
    """Pair each agent with the `band` agents either side of it by rating."""
    if band < 1:
        raise ValueError("band must be at least 1")

    def _neighbours(pool: Sequence[AgentInfo]) -> Sequence[tuple[UUID, UUID]]:
        ranked = sorted(pool, key=lambda info: info.elo)

        return [
            (ranked[lower].id, ranked[upper].id)
            for lower in range(len(ranked))
            for upper in range(lower + 1, min(lower + band + 1, len(ranked)))
        ]

    return _neighbours
