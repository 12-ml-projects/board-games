from typing import Protocol, TypeAlias

import pytest

from src.ai.minimax import MiniMax
from src.game.game import Game, Move, State
from src.tournament.bradley_terry import bradley_terry
from src.tournament.round_robin import round_robin
from src.tournament.tournament import Tournament

Seed: TypeAlias = int
Nodes: TypeAlias = int


class MiniMaxAgentsFactory[StateT: State, MoveT: Move](Protocol):
    def __call__(
        self, game: Game[StateT, MoveT], hyperparams: list[tuple[Nodes, Seed]]
    ) -> list[MiniMax[StateT, MoveT]]: ...


class MinimaxTournamentFactory[StateT: State, MoveT: Move](Protocol):
    def __call__(
        self,
        game: Game[StateT, MoveT],
        hyperparams: list[tuple[Nodes, Seed]],
        num_rounds: int,
    ) -> Tournament[StateT, MoveT]: ...


@pytest.fixture()
def minimax_agents() -> MiniMaxAgentsFactory:
    def _minimax_agents(
        game: Game, hyperparams: list[tuple[Nodes, Seed]]
    ) -> list[MiniMax]:
        return [
            MiniMax(game=game, max_nodes=max_nodes, seed=seed)
            for max_nodes, seed in hyperparams
        ]

    return _minimax_agents


@pytest.fixture()
def minimax_tournament(
    minimax_agents: MiniMaxAgentsFactory,
) -> MinimaxTournamentFactory:
    def _minimax_tournament(
        game: Game, hyperparams: list[tuple[Nodes, Seed]], num_rounds: int
    ) -> Tournament:
        return Tournament(
            game=game,
            match_maker=round_robin,
            elo_estimator=bradley_terry,
            agents=minimax_agents(game=game, hyperparams=hyperparams),
            num_rounds=num_rounds,
        )

    return _minimax_tournament
