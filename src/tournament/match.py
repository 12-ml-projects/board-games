from dataclasses import dataclass
from typing import Sequence, TypeAlias
from uuid import UUID

from src.ai.ai import AI
from src.exceptions import IllegalMoveException, MatchTooLongException
from src.game import Game, Move, Outcome, State
from src.player import Player

Round: TypeAlias = int


@dataclass(frozen=True, kw_only=True)
class MatchResult[StateT: State, MoveT: Move]:
    outcome: Outcome
    states: Sequence[StateT]
    moves: Sequence[MoveT]


# Which agents met, in which round, and how it went. The pairing is ordered:
# the first agent played as A.
MatchRecord: TypeAlias = tuple[Round, tuple[UUID, UUID], MatchResult]


class Match[StateT: State, MoveT: Move]:
    _agent_a: AI[StateT, MoveT]
    _agent_b: AI[StateT, MoveT]
    _game: Game[StateT, MoveT]

    def __init__(
        self,
        agent_a: AI[StateT, MoveT],
        agent_b: AI[StateT, MoveT],
        game: Game[StateT, MoveT],
    ):
        self._game = game
        self._agent_a = agent_a
        self._agent_b = agent_b

    def _get_agent(self, player: Player) -> AI[StateT, MoveT]:
        if player == "A":
            return self._agent_a
        elif player == "B":
            return self._agent_b
        else:
            raise ValueError(f"Unknown player: {player}")

    def play(self, max_plies: int = 200) -> MatchResult[StateT, MoveT]:
        states: list[StateT] = [self._game.initial()]
        moves: list[MoveT] = []

        while True:
            current_state = states[-1]

            if self._game.is_final_state(current_state):
                break

            if len(states) > max_plies:
                raise MatchTooLongException(
                    f"Match exceeded the max allowed number of plies: {max_plies}"
                )

            agent = self._get_agent(current_state.player)
            move = agent.choose_move(current_state)
            if not self._game.is_legal_move(current_state, move):
                raise IllegalMoveException(f"Illegal move: {move}")

            current_state = self._game.make_move(current_state, move)

            moves.append(move)
            states.append(current_state)

        return MatchResult(
            outcome=self._game.outcome(states[-1]),
            states=tuple(states),
            moves=tuple(moves),
        )
