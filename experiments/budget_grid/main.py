"""Every budget against every other, at each noise setting."""

from typing import Sequence

import click

from experiments.games import GAMES
from experiments.sweep import PairResult, sweep, total_games, write_results


@click.command()
@click.option("--game-name", type=click.Choice(sorted(GAMES)), default="connect_four")
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
    default=(0.0, 0.04, 0.08),
    help="Noise settings to try; every agent carries the same one.",
)
@click.option("--games", type=int, default=50, help="Games played per pairing.")
@click.option("--seed", type=int, default=0)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write results as JSON.",
)
def budget_grid(
    game_name: str,
    budgets: Sequence[int],
    noise_levels: Sequence[float],
    games: int,
    seed: int,
    out: str | None,
) -> None:
    game = GAMES[game_name]()
    ranked = sorted(set(budgets))

    played = total_games(ranked, noise_levels, games)

    try:
        with click.progressbar(length=played, label="playing") as progress:
            results = sweep(
                game,
                budgets=ranked,
                noise_levels=noise_levels,
                games=games,
                seed=seed,
                on_game=lambda: progress.update(1),
            )
    except ValueError as invalid:
        raise click.BadParameter(str(invalid))

    click.echo(
        f"\n{game_name}: {len(ranked)} budgets, {len(results)} pairings played, "
        f"{played} games in total"
    )

    for noise in noise_levels:
        _show_matrix(ranked, [r for r in results if r.noise == noise], noise)

    if out:
        write_results(out, game_name=game_name, seed=seed, results=results)
        click.echo(f"\nwritten to {out}")


def _show_matrix(
    ranked: Sequence[int], results: Sequence[PairResult], noise: float
) -> None:
    shares = {}

    for result in results:
        shares[(result.stronger, result.weaker)] = result.score
        shares[(result.weaker, result.stronger)] = 1.0 - result.score

    click.echo(f"\nnoise = {noise}   (row's score share against column)")
    click.echo("        | " + " ".join(f"{budget:>6}" for budget in ranked))

    for row in ranked:
        cells = " ".join(
            "     -" if row == column else f"{shares[(row, column)]:>6.2f}"
            for column in ranked
        )
        click.echo(f"  {row:>5} | {cells}")


if __name__ == "__main__":
    budget_grid()
