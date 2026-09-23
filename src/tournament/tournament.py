import uuid
from typing import Sequence
from uuid import UUID

from src.ai.ai import AI
from src.game import Game, Move, State
from src.tournament.agent_info import AgentInfo
from src.tournament.elo import BASELINE_ELO, EloEstimator
from src.tournament.match import Match, MatchRecord
from src.tournament.match_maker import MatchMaker


class Tournament[StateT: State, MoveT: Move]:
    _match_maker: MatchMaker
    _elo_estimator: EloEstimator
    _num_rounds: int
    _game: Game[StateT, MoveT]

    _agents: dict[UUID, AI[StateT, MoveT]]
    _agent_infos: dict[UUID, AgentInfo]

    def __init__(
        self,
        match_maker: MatchMaker,
        elo_estimator: EloEstimator,
        agents: Sequence[AI[StateT, MoveT]],
        game: Game[StateT, MoveT],
        num_rounds: int,
    ):
        self._match_maker = match_maker
        self._elo_estimator = elo_estimator
        self._num_rounds = num_rounds
        self._game = game

        self._agents = {}
        self._agent_infos = {}

        for agent in agents:
            self.add_agent(agent)

    def add_agent(self, agent: AI[StateT, MoveT]) -> UUID:
        """Enter an agent, unrated, to be rated by the next `run`."""
        if any(a.description == agent.description for a in self._agents.values()):
            raise ValueError(
                f"An agent already describes itself as {agent.description}"
            )

        agent_id = uuid.uuid4()

        self._agents[agent_id] = agent
        self._agent_infos[agent_id] = AgentInfo(id=agent_id, elo=BASELINE_ELO)

        return agent_id

    @property
    def standings(self) -> Sequence[AgentInfo]:
        return sorted(self._agent_infos.values(), key=lambda info: -info.elo)

    def run(self) -> Sequence[MatchRecord]:
        records: list[MatchRecord] = []

        for round_number in range(self._num_rounds):
            pool = list(self._agent_infos.values())

            for pairing in self._match_maker(pool):
                records.extend(self._play_both_ways(round_number, pairing))

            # Re-rate between rounds, so that a match maker pairing on rating
            # has something to pair on.
            self._assign_elos(records)

        return records

    def _play_both_ways(
        self, round_number: int, pairing: tuple[UUID, UUID]
    ) -> list[MatchRecord]:
        """Play a pairing twice, once from each side."""
        first, second = pairing
        records: list[MatchRecord] = []

        for id_a, id_b in ((first, second), (second, first)):
            match = Match[StateT, MoveT](
                agent_a=self._agents[id_a],
                agent_b=self._agents[id_b],
                game=self._game,
            )

            records.append((round_number, (id_a, id_b), match.play()))

        return records

    def _assign_elos(self, records: Sequence[MatchRecord]) -> None:
        elos = self._elo_estimator(records)

        self._agent_infos = {
            agent_id: AgentInfo(id=agent_id, elo=elos.get(agent_id, BASELINE_ELO))
            for agent_id in self._agents
        }

    def get_by_description(
        self, description: str
    ) -> tuple[AI[StateT, MoveT], AgentInfo]:
        ais = [
            agent for agent in self._agents.values() if agent.description == description
        ]

        if len(ais) == 0:
            raise ValueError(f"No agents found with description '{description}'")

        if len(ais) > 1:
            raise ValueError(f"Multiple agents found with description '{description}'")

        agent = ais[0]
        agent_id = next(id for id, a in self._agents.items() if a == agent)

        return agent, self._agent_infos[agent_id]
