from typing import Protocol, Sequence
from uuid import UUID

from src.tournament.agent_info import AgentInfo


class MatchMaker(Protocol):
    def __call__(self, pool: Sequence[AgentInfo]) -> Sequence[tuple[UUID, UUID]]: ...
