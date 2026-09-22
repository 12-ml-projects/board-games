from abc import ABC
from dataclasses import dataclass

from src.player import Player


@dataclass(frozen=True, slots=True, kw_only=True)
class State(ABC):
    player: Player
