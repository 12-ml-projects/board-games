from dataclasses import dataclass
from typing import Protocol, Sequence

from src.ai.ai import AI
from src.ai.mcts import MCTS
from src.ai.minimax import MiniMax
from src.game.game import Game
from src.game.move import Move
from src.game.outcome import SAFE_MARGIN
from src.game.state import State


class AgentFactory[StateT: State, MoveT: Move](Protocol):
    def __call__(
        self, game: Game[StateT, MoveT], *, budget: int, seed: int
    ) -> AI[StateT, MoveT]: ...


@dataclass(frozen=True, kw_only=True)
class Variant:
    """One agent configuration, to be swept across a range of budgets."""

    family: str
    label: str
    build: AgentFactory


def minimax_variants(noise_levels: Sequence[float]) -> list[Variant]:
    if any(noise < 0.0 or noise >= SAFE_MARGIN for noise in noise_levels):
        raise ValueError(f"all noise levels must be at least 0 and below {SAFE_MARGIN}")

    return [
        Variant(family="minimax", label=f"noise={noise}", build=_minimax(noise))
        for noise in noise_levels
    ]


def mcts_variants(explorations: Sequence[float]) -> list[Variant]:
    if any(exploration < 0.0 for exploration in explorations):
        raise ValueError("all exploration constants must be non-negative")

    return [
        Variant(
            family="mcts", label=f"exploration={exploration}", build=_mcts(exploration)
        )
        for exploration in explorations
    ]


def _minimax(noise: float) -> AgentFactory:
    def build(game: Game, *, budget: int, seed: int) -> AI:
        return MiniMax(game, max_nodes=budget, noise=noise, seed=seed)

    return build


def _mcts(exploration: float) -> AgentFactory:
    def build(game: Game, *, budget: int, seed: int) -> AI:
        return MCTS(game, n_simulations=budget, exploration=exploration, seed=seed)

    return build
