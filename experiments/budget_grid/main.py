"""Every budget of one agent family against every other, at each setting."""

from typing import Sequence

import click

from experiments.games import GAMES
from experiments.options import shared_options
from experiments.sweep import PairResult, sweep, total_games, write_results
from experiments.variants import Variant, mcts_variants, minimax_variants


@click.group()
def budget_grid() -> None:
    pass


@budget_grid.command("minimax")
@shared_options
@click.option(
    "--nodes",
    "budgets",
    type=int,
    multiple=True,
    default=(50, 200, 800, 3200),
    help="Node budgets to pit against one another.",
)
@click.option(
    "--noise",
    "noise_levels",
    type=float,
    multiple=True,
    default=(0.0,),
    help="Noise settings to try; every agent carries the same one.",
)
def minimax_command(
    game_name: str,
    games: int,
    seed: int,
    out: str | None,
    budgets: Sequence[int],
    noise_levels: Sequence[float],
) -> None:
    """Minimax, budgeted by nodes searched."""
    _report(
        game_name=game_name,
        budgets=budgets,
        variants=minimax_variants(noise_levels),
        games=games,
        seed=seed,
        out=out,
    )


@budget_grid.command("mcts")
@shared_options
@click.option(
    "--simulations",
    "budgets",
    type=int,
    multiple=True,
    default=(20, 50, 200, 800),
    help="Simulation budgets to pit against one another.",
)
@click.option(
    "--exploration",
    "explorations",
    type=float,
    multiple=True,
    default=(1.4,),
    help="Exploration constants to try; every agent carries the same one.",
)
def mcts_command(
    game_name: str,
    games: int,
    seed: int,
    out: str | None,
    budgets: Sequence[int],
    explorations: Sequence[float],
) -> None:
    """MCTS, budgeted by simulations run."""
    _report(
        game_name=game_name,
        budgets=budgets,
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
    ranked = sorted(set(budgets))

    with click.progressbar(
        length=total_games(ranked, variants, games), label="playing"
    ) as progress:
        results = sweep(
            game,
            variants=variants,
            budgets=ranked,
            games=games,
            seed=seed,
            on_game=lambda: progress.update(1),
        )

    click.echo(
        f"\n{game_name}: {len(ranked)} budgets, {len(results)} pairings played, "
        f"{games} games each"
    )

    for variant in variants:
        _show_matrix(
            ranked, [r for r in results if r.variant == variant.label], variant
        )

    if out:
        write_results(out, game_name=game_name, seed=seed, results=results)
        click.echo(f"\nwritten to {out}")


def _show_matrix(
    ranked: Sequence[int], results: Sequence[PairResult], variant: Variant
) -> None:
    shares = {}

    for result in results:
        shares[(result.stronger, result.weaker)] = result.score
        shares[(result.weaker, result.stronger)] = 1.0 - result.score

    click.echo(f"\n{variant.label}   (row's score share against column)")
    click.echo("        | " + " ".join(f"{budget:>6}" for budget in ranked))

    for row in ranked:
        cells = " ".join(
            "     -" if row == column else f"{shares[(row, column)]:>6.2f}"
            for column in ranked
        )
        click.echo(f"  {row:>5} | {cells}")


if __name__ == "__main__":
    budget_grid()
