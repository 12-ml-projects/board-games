"""How far agents can be perturbed before their budgets stop separating them.

Two budgets of one agent family are played against each other under each of
the family's settings, so you can see how much disturbance the gap survives.
"""

from typing import Sequence

import click

from experiments.games import GAMES
from experiments.options import shared_options
from experiments.sweep import sweep, total_games, write_results
from experiments.variants import Variant, mcts_variants, minimax_variants


@click.group()
def noise_sensitivity() -> None:
    """Compare two budgets of one agent family across its settings."""


@noise_sensitivity.command("minimax")
@shared_options
@click.option("--weak-nodes", type=int, default=150, help="Budget of the weaker agent.")
@click.option(
    "--strong-nodes", type=int, default=1500, help="Budget of the stronger agent."
)
@click.option(
    "--noise",
    "noise_levels",
    type=float,
    multiple=True,
    default=(0.0, 0.01, 0.02, 0.04, 0.06, 0.08),
    help="Noise settings to try; both agents carry the same one.",
)
def minimax_command(
    game_name: str,
    games: int,
    seed: int,
    out: str | None,
    weak_nodes: int,
    strong_nodes: int,
    noise_levels: Sequence[float],
) -> None:
    """Minimax, budgeted by nodes searched, swept over noise."""
    _report(
        game_name=game_name,
        budgets=(weak_nodes, strong_nodes),
        variants=minimax_variants(noise_levels),
        games=games,
        seed=seed,
        out=out,
    )


@noise_sensitivity.command("mcts")
@shared_options
@click.option(
    "--weak-simulations", type=int, default=20, help="Budget of the weaker agent."
)
@click.option(
    "--strong-simulations", type=int, default=200, help="Budget of the stronger agent."
)
@click.option(
    "--exploration",
    "explorations",
    type=float,
    multiple=True,
    default=(0.5, 1.0, 1.4, 2.0, 3.0),
    help="Exploration constants to try; both agents carry the same one.",
)
def mcts_command(
    game_name: str,
    games: int,
    seed: int,
    out: str | None,
    weak_simulations: int,
    strong_simulations: int,
    explorations: Sequence[float],
) -> None:
    """MCTS, budgeted by simulations run, swept over the exploration constant."""
    _report(
        game_name=game_name,
        budgets=(weak_simulations, strong_simulations),
        variants=mcts_variants(explorations),
        games=games,
        seed=seed,
        out=out,
    )


def _report(
    *,
    game_name: str,
    budgets: Sequence[int],
    variants: Sequence[Variant],
    games: int,
    seed: int,
    out: str | None,
) -> None:
    game = GAMES[game_name]()
    weaker, stronger = sorted(budgets)

    with click.progressbar(
        length=total_games(budgets, variants, games), label="playing"
    ) as progress:
        results = sweep(
            game,
            variants=variants,
            budgets=budgets,
            games=games,
            seed=seed,
            on_game=lambda: progress.update(1),
        )

    click.echo(
        f"\n{game_name}: {stronger} against {weaker}, {games} games per setting\n"
    )
    click.echo("  setting              | strong  weak  draw | score | 95% interval")

    for result in results:
        click.echo(
            f"  {result.variant:<20} | {result.strong_wins:>6} "
            f"{result.weak_wins:>5} {result.draws:>5} | {result.score:5.2f} "
            f"| [{result.low:4.2f}, {result.high:4.2f}]"
        )

    separating = [r.variant for r in results if r.separated]
    click.echo(f"\nseparated at: {', '.join(separating) or 'no setting'}")

    if out:
        write_results(out, game_name=game_name, seed=seed, results=results)
        click.echo(f"written to {out}")


if __name__ == "__main__":
    noise_sensitivity()
