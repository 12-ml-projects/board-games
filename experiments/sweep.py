import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Callable, Sequence

from experiments.variants import Variant
from experiments.wilson import wilson_interval
from src.game.game import Game
from src.game.outcome import DRAW, ONGOING
from src.tournament.match import Match


@dataclass(frozen=True, kw_only=True)
class PairResult:
    variant: str
    stronger: int
    weaker: int
    games: int
    strong_wins: int
    weak_wins: int
    draws: int
    low: float
    high: float

    @property
    def score(self) -> float:
        """Points to the agent with the larger budget, a draw being half."""
        return (self.strong_wins + self.draws / 2) / self.games

    @property
    def separated(self) -> bool:
        """Whether the larger budget is ahead by more than the interval."""
        return self.low > 0.5


def pairings(budgets: Sequence[int]) -> list[tuple[int, int]]:
    """Every budget paired with every other, weaker first."""
    return list(combinations(sorted(set(budgets)), 2))


def total_games(budgets: Sequence[int], variants: Sequence[Variant], games: int) -> int:
    return len(pairings(budgets)) * len(variants) * games


def play_pair(
    game: Game,
    *,
    variant: Variant,
    stronger: int,
    weaker: int,
    games: int,
    seed: int,
    on_game: Callable[[], None] | None = None,
) -> PairResult:
    strong_agent = variant.build(game, budget=stronger, seed=seed)
    weak_agent = variant.build(game, budget=weaker, seed=seed + 1)

    strong_wins = 0
    weak_wins = 0
    draws = 0

    for index in range(games):
        stronger_is_first = index % 2 == 0
        first, second = (
            (strong_agent, weak_agent)
            if stronger_is_first
            else (weak_agent, strong_agent)
        )

        outcome = Match(agent_a=first, agent_b=second, game=game).play().outcome

        if outcome == ONGOING:
            raise RuntimeError("a match came back unfinished")

        if outcome == DRAW:
            draws += 1
        elif (outcome == "A") == stronger_is_first:
            strong_wins += 1
        else:
            weak_wins += 1

        if on_game:
            on_game()

    low, high = wilson_interval((strong_wins + draws / 2) / games, games)

    return PairResult(
        variant=variant.label,
        stronger=stronger,
        weaker=weaker,
        games=games,
        strong_wins=strong_wins,
        weak_wins=weak_wins,
        draws=draws,
        low=low,
        high=high,
    )


def sweep(
    game: Game,
    *,
    variants: Sequence[Variant],
    budgets: Sequence[int],
    games: int,
    seed: int,
    on_game: Callable[[], None] | None = None,
) -> list[PairResult]:
    """Every budget against every other, under every variant."""

    order = {variant.label: position for position, variant in enumerate(variants)}
    work = [
        (variant, weaker, stronger)
        for variant in variants
        for weaker, stronger in pairings(budgets)
    ]
    random.Random(seed).shuffle(work)

    results = [
        play_pair(
            game,
            variant=variant,
            stronger=stronger,
            weaker=weaker,
            games=games,
            seed=seed,
            on_game=on_game,
        )
        for variant, weaker, stronger in work
    ]

    return sorted(results, key=lambda r: (order[r.variant], r.weaker, r.stronger))


def write_results(
    path: str, *, game_name: str, seed: int, results: Sequence[PairResult]
) -> None:
    with open(path, "w") as handle:
        json.dump(
            {
                "game": game_name,
                "seed": seed,
                "results": [{**asdict(r), "score": r.score} for r in results],
            },
            handle,
            indent=2,
        )
