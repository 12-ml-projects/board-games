from math import tanh
from typing import Final, TypeAlias

from src.game.game import Game
from src.game.outcome import DRAW, HEURISTIC_CEILING, ONGOING, Outcome
from src.player import Player, opposite_player
from src.tic_tac_toe.move import TTTMove
from src.tic_tac_toe.state import Board, Cell, TTTState

Line: TypeAlias = tuple[tuple[int, int], ...]


LINES: Final[tuple[Line, ...]] = (
    *(tuple((x, y) for y in range(3)) for x in range(3)),
    *(tuple((x, y) for x in range(3)) for y in range(3)),
    tuple((i, i) for i in range(3)),
    tuple((i, 2 - i) for i in range(3)),
)


class TTTGame(Game[TTTState, TTTMove]):
    def initial(self) -> TTTState:
        board: Board = tuple(tuple("" for _ in range(3)) for _ in range(3))

        return TTTState(board=board, player="A")

    def legal_moves(self, state: TTTState) -> list[TTTMove]:
        available_cells = [
            (x, y) for x in range(3) for y in range(3) if state.board[x][y] == ""
        ]

        return [TTTMove(x=x, y=y) for x, y in available_cells]

    def make_move(self, state: TTTState, move: TTTMove) -> TTTState:
        symbol = self._get_symbol_for_player(state.player)
        new_board = tuple(
            tuple(
                symbol if (x, y) == (move.x, move.y) else cell
                for y, cell in enumerate(row)
            )
            for x, row in enumerate(state.board)
        )

        return TTTState(board=new_board, player=opposite_player(state.player))

    def outcome(self, state: TTTState) -> Outcome:
        board = state.board

        for line in LINES:
            first, second, third = (board[x][y] for x, y in line)
            if first != "" and first == second == third:
                return self._get_player_for_symbol(first)

        if all(cell != "" for row in board for cell in row):
            return DRAW

        return ONGOING

    def material(self, state: TTTState) -> float:
        """Counts winnable lines."""
        mine = self._get_symbol_for_player(state.player)
        theirs = self._get_symbol_for_player(opposite_player(state.player))

        return float(
            self._open_lines(state.board, theirs) - self._open_lines(state.board, mine)
        )

    def evaluate_material(self, material_advantage: float) -> float:
        return tanh(material_advantage) * HEURISTIC_CEILING

    def _open_lines(self, board: Board, blocked_by: Cell) -> int:
        return sum(
            1 for line in LINES if all(board[x][y] != blocked_by for x, y in line)
        )

    def _get_symbol_for_player(self, player: Player) -> Cell:
        if player == "A":
            return "x"
        elif player == "B":
            return "o"

        raise ValueError(f"Unknown player: {player}")

    def _get_player_for_symbol(self, symbol: Cell) -> Player:
        if symbol == "x":
            return "A"
        elif symbol == "o":
            return "B"

        raise ValueError(f"No player for symbol: {symbol!r}")
