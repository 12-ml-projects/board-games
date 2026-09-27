from dataclasses import dataclass, field

from src.game import Move, State


@dataclass(slots=True)
class Node[StateT: State, MoveT: Move]:
    state: StateT
    prior: float
    visits: int = 0
    total_value: float = 0.0
    children: dict[MoveT, "Node[StateT, MoveT]"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)
