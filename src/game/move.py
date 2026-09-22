from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class Move(ABC):
    pass
