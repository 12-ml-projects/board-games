"""Options every experiment command takes."""

from typing import Callable, TypeVar

import click

from experiments.games import GAMES

Command = TypeVar("Command", bound=Callable)

SHARED = (
    click.option(
        "--game-name", type=click.Choice(sorted(GAMES)), default="connect_four"
    ),
    click.option("--games", type=int, default=40, help="Games played per pairing."),
    click.option("--seed", type=int, default=0),
    click.option(
        "--out",
        type=click.Path(dir_okay=False),
        default=None,
        help="Write results as JSON.",
    ),
)


def shared_options(command: Command) -> Command:
    for option in reversed(SHARED):
        command = option(command)

    return command
