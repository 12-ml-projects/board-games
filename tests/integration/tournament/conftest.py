from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import pytest

from experiments.variants import Variant, mcts_variants, minimax_variants
from src.game.game import Game, Move, State
from src.tournament.bradley_terry import bradley_terry
from src.tournament.round_robin import round_robin
from src.tournament.tournament import Tournament

# What a budget buys differs per family, so every test names its budgets per
# family. Five times apart is enough for either to separate on either game
# without the suite taking too long.
BUDGETS: Mapping[str, Sequence[int]] = {
    "minimax": (5, 25, 125),
    "mcts": (5, 25, 125),
}


@dataclass(frozen=True, kw_only=True)
class Ladder[StateT: State, MoveT: Move]:
    """A tournament, plus how its agents describe themselves, weakest first.

    Agents name themselves differently per family, so a test that wants the
    rungs in order asks here rather than spelling out descriptions itself.
    """

    tournament: Tournament[StateT, MoveT]
    descriptions: Sequence[str]

    def elos(self) -> list[float]:
        return [
            self.tournament.get_by_description(description)[1].elo
            for description in self.descriptions
        ]


class LadderFactory[StateT: State, MoveT: Move](Protocol):
    def __call__(
        self,
        game: Game[StateT, MoveT],
        budgets: Mapping[str, Sequence[int]],
        num_rounds: int,
    ) -> Ladder[StateT, MoveT]: ...


@pytest.fixture(params=["ttt", "c4"])
def game(request: pytest.FixtureRequest) -> Game:
    """Each test using this runs once per game."""
    return request.getfixturevalue(request.param)()


@pytest.fixture(params=["minimax", "mcts"])
def variant(request: pytest.FixtureRequest) -> Variant:
    """Each test using this runs once per agent family."""
    families = {
        "minimax": lambda: minimax_variants([0.0]),
        "mcts": lambda: mcts_variants([1.4]),
    }

    return families[request.param]()[0]


@pytest.fixture()
def ladder(variant: Variant) -> LadderFactory:
    """Build a tournament of one family's agents, at that family's budgets."""

    def _ladder(
        game: Game, budgets: Mapping[str, Sequence[int]], num_rounds: int
    ) -> Ladder:
        agents = [
            variant.build(game, budget=budget, seed=seed)
            for seed, budget in enumerate(budgets[variant.family])
        ]

        return Ladder(
            tournament=Tournament(
                game=game,
                match_maker=round_robin,
                elo_estimator=bradley_terry,
                agents=agents,
                num_rounds=num_rounds,
            ),
            descriptions=[agent.description for agent in agents],
        )

    return _ladder
