"""How much noise can agents carry before their budgets stop telling them apart?

Noise is what keeps two agents from replaying one game, but it also blurs the
gap the ladder is supposed to measure. This pitches one budget against another
across a range of noise settings and reports how clearly the stronger one still
comes out ahead.
"""

from typing import Sequence

import click

from experiments.games import GAMES
from experiments.sweep import sweep, total_games, write_results


@click.command()
@click.option("--game-name", type=click.Choice(sorted(GAMES)), default="connect_four")
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
@click.option("--games", type=int, default=40, help="Games played at each setting.")
@click.option("--seed", type=int, default=0)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write results as JSON.",
)
def noise_sensitivity(
    game_name: str,
    weak_nodes: int,
    strong_nodes: int,
    noise_levels: Sequence[float],
    games: int,
    seed: int,
    out: str | None,
) -> None:
    game = GAMES[game_name]()

    budgets = (weak_nodes, strong_nodes)

    try:
        with click.progressbar(
            length=total_games(budgets, noise_levels, games), label="playing"
        ) as progress:
            results = sweep(
                game,
                budgets=budgets,
                noise_levels=noise_levels,
                games=games,
                seed=seed,
                on_game=lambda: progress.update(1),
            )
    except ValueError as invalid:
        raise click.BadParameter(str(invalid))

    click.echo(
        f"\n{game_name}: minimax-{strong_nodes} against minimax-{weak_nodes}, "
        f"{games} games per setting\n"
    )
    click.echo("  noise | strong  weak  draw | score | 95% interval")

    for result in results:
        click.echo(
            f"  {result.noise:>5} | {result.strong_wins:>6} {result.weak_wins:>5} "
            f"{result.draws:>5} | {result.score:5.2f} "
            f"| [{result.low:4.2f}, {result.high:4.2f}]"
        )

    separating = [r.noise for r in results if r.separated]
    click.echo(f"\nclearly separated up to noise={max(separating, default=None)}")

    if out:
        write_results(out, game_name=game_name, seed=seed, results=results)
        click.echo(f"written to {out}")


if __name__ == "__main__":
    noise_sensitivity()
