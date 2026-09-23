from itertools import combinations
from typing import Sequence
from uuid import UUID

from src.tournament.agent_info import AgentInfo


def round_robin(pool: Sequence[AgentInfo]) -> Sequence[tuple[UUID, UUID]]:
    return [(a.id, b.id) for a, b in combinations(pool, 2)]
