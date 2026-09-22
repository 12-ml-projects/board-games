from typing import Literal, TypeAlias

Player: TypeAlias = Literal["A", "B"]


def opposite_player(player: Player) -> Player:
    return "B" if player == "A" else "A"
