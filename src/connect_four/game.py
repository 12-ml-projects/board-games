from functools import cached_property
from itertools import product
from math import tanh
from typing import Final, TypeAlias

from src.connect_four.exceptions import ColumnIsFullException
from src.connect_four.move import C4Move
from src.connect_four.state import Board, C4State, Cell
from src.game.game import Game
from src.game.outcome import DRAW, ONGOING, Outcome
from src.player import Player, opposite_player

Position: TypeAlias = tuple[int, int]
Window: TypeAlias = tuple[Position, ...]


# Cap for non-winning positions
HEURISTIC_CEILING: Final[float] = 0.9


# Running window length needed to win
CONNECT: Final[int] = 4


# Directions in which a window can extend
DIRECTIONS: Final[tuple[Position, ...]] = ((0, 1), (1, 0), (1, 1), (1, -1))


# What a window of CONNECT cells is worth when it holds this many of a cell type
WINDOW_VALUE: Final[dict[int, float]] = {2: 1.0, 3: 4.0}


# A piece in the middle column sits on more potential lines than at the edge
CENTRE_VALUE: Final[float] = 0.5


# Scaler for the tanh function. Avoid saturating near the tails
ADVANTAGE_SCALE: Final[float] = 8.0


class C4Game(Game[C4State, C4Move]):
    _width: int
    _height: int

    def __init__(self, width: int, height: int):
        if width < CONNECT:
            raise ValueError(f"Width must be at least {CONNECT}")

        if height < CONNECT:
            raise ValueError(f"Height must be at least {CONNECT}")

        self._width = width
        self._height = height

    def initial(self) -> C4State:
        board: Board = tuple(
            tuple("" for _ in range(self._width)) for _ in range(self._height)
        )

        return C4State(board=board, player="A")

    def legal_moves(self, state: C4State) -> list[C4Move]:
        moves: list[C4Move] = []
        for col_idx in range(self._width):
            if self._is_full(col_idx, state):
                continue

            if self._first_open_row(col_idx, state) != -1:
                moves.append(C4Move(column=col_idx))

        return moves

    def make_move(self, state: C4State, move: C4Move) -> C4State:
        symbol = self._get_symbol_for_player(state.player)

        row_idx = self._first_open_row(move.column, state)

        if row_idx == -1:
            raise ColumnIsFullException(f"Column {move.column} is full")

        board = [list(row) for row in state.board]
        board[row_idx][move.column] = symbol
        new_board: Board = tuple(tuple(row) for row in board)

        return C4State(board=new_board, player=opposite_player(state.player))

    def is_legal_move(self, state: C4State, move: C4Move) -> bool:
        if not 0 <= move.column < self._width:
            return False

        return not self._is_full(move.column, state)

    def outcome(self, state: C4State) -> Outcome:
        board = state.board

        for row_idx, col_idx in product(range(self._height), range(self._width)):
            symbol = board[row_idx][col_idx]

            if symbol == "":
                continue

            if any(self._is_line(board, row_idx, col_idx, step) for step in DIRECTIONS):
                return self._get_player_for_symbol(symbol)

        if self._is_board_full(board):
            return DRAW

        return ONGOING

    def material(self, state: C4State) -> float:
        """How much better the board looks for `state.player`."""
        mine = self._get_symbol_for_player(state.player)
        theirs = self._get_symbol_for_player(opposite_player(state.player))
        board = state.board

        value = 0.0

        for window in self._window_cells:
            held = 0
            opposed = 0

            for row_idx, col_idx in window:
                cell = board[row_idx][col_idx]

                if cell == mine:
                    held += 1
                elif cell == theirs:
                    opposed += 1

            if opposed == 0:
                value += WINDOW_VALUE.get(held, 0.0)
            elif held == 0:
                value -= WINDOW_VALUE.get(opposed, 0.0)

        return value + self._centre_hold(board, mine, theirs)

    def evaluate_material(self, material_advantage: float) -> float:
        return tanh(material_advantage / ADVANTAGE_SCALE) * HEURISTIC_CEILING

    def _centre_hold(self, board: Board, mine: Cell, theirs: Cell) -> float:
        centre = self._width // 2
        held = sum(1 for row in board if row[centre] == mine)
        opposed = sum(1 for row in board if row[centre] == theirs)

        return (held - opposed) * CENTRE_VALUE

    @cached_property
    def _window_cells(self) -> tuple[Window, ...]:
        """Every run of CONNECT squares that fits, in all four directions.

        These depend only on the board's shape, and material() walks all of
        them at every node, so they are worked out once per game.
        """
        return tuple(
            tuple(window)
            for row_idx, col_idx in product(range(self._height), range(self._width))
            for step in DIRECTIONS
            if (window := self._line_cells(row_idx, col_idx, step)) is not None
        )

    def _is_board_full(self, board: Board) -> bool:
        # Pieces stack from the bottom, so a full top row means a full board.
        return all(cell != "" for cell in board[self._height - 1])

    def _is_line(
        self, board: Board, row_idx: int, col_idx: int, step: Position
    ) -> bool:
        cells = self._line_cells(row_idx, col_idx, step)

        if cells is None:
            return False

        symbol = board[row_idx][col_idx]

        return all(board[row][col] == symbol for row, col in cells)

    def _line_cells(
        self, row_idx: int, col_idx: int, step: Position
    ) -> list[Position] | None:
        row_step, col_step = step
        cells = [
            (row_idx + i * row_step, col_idx + i * col_step) for i in range(CONNECT)
        ]

        if any(
            not 0 <= row < self._height or not 0 <= col < self._width
            for row, col in cells
        ):
            return None

        return cells

    def _is_full(self, col_idx: int, state: C4State) -> bool:
        column = [row[col_idx] for row in state.board]

        return all([cell in ["x", "o"] for cell in column])

    def _first_open_row(self, col_idx: int, state: C4State) -> int:
        for row_idx in range(self._height):
            if state.board[row_idx][col_idx] == "":
                return row_idx

        return -1

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
