"""Playing budgets off against each other, at given noise settings.

Both experiments ask the same question of the same machinery and differ only
in how they arrange the answer, so the playing lives here and the commands
are left to present it.
"""

import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Callable, Sequence

from experiments.wilson import wilson_interval
from src.ai.minimax import MiniMax
from src.game.game import Game
from src.game.outcome import DRAW, ONGOING, SAFE_MARGIN
from src.tournament.match import Match


@dataclass(frozen=True, kw_only=True)
class PairResult:
    stronger: int
    weaker: int
    noise: float
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


def total_games(
    budgets: Sequence[int], noise_levels: Sequence[float], games: int
) -> int:
    return len(pairings(budgets)) * len(noise_levels) * games


def play_pair(
    game: Game,
    *,
    stronger: int,
    weaker: int,
    noise: float,
    games: int,
    seed: int,
    on_game: Callable[[], None] | None = None,
) -> PairResult:
    if not 0.0 <= noise < SAFE_MARGIN:
        raise ValueError(f"noise must be at least 0 and below {SAFE_MARGIN}")

    strong_agent = MiniMax(game, max_nodes=stronger, noise=noise, seed=seed)
    weak_agent = MiniMax(game, max_nodes=weaker, noise=noise, seed=seed + 1)

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
        stronger=stronger,
        weaker=weaker,
        noise=noise,
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
    budgets: Sequence[int],
    noise_levels: Sequence[float],
    games: int,
    seed: int,
    on_game: Callable[[], None] | None = None,
) -> list[PairResult]:
    order = {noise: position for position, noise in enumerate(noise_levels)}
    work = [
        (noise, weaker, stronger)
        for noise in noise_levels
        for weaker, stronger in pairings(budgets)
    ]
    random.Random(seed).shuffle(work)

    results = [
        play_pair(
            game,
            stronger=stronger,
            weaker=weaker,
            noise=noise,
            games=games,
            seed=seed,
            on_game=on_game,
        )
        for noise, weaker, stronger in work
    ]

    return sorted(results, key=lambda r: (order[r.noise], r.weaker, r.stronger))


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
