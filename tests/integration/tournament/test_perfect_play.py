from typing import Mapping, Sequence

from src.ai.random import RandomAI
from src.game.outcome import DRAW
from tests.integration.conftest import TTTFactory
from tests.integration.tournament.conftest import LadderFactory

# Tic-tac-toe is a draw under perfect play, so an agent that searches it out
# exhaustively can neither lose nor beat a copy of itself. Both families reach
# that point, but at budgets an order of magnitude apart.
PERFECT: Mapping[str, Sequence[int]] = {
    "minimax": (10_000, 10_001),
    "mcts": (500, 501),
}


class TestTicTacToePerfectPlay:
    def test_perfect_play_draws_against_itself(
        self, ladder: LadderFactory, ttt: TTTFactory
    ):
        rungs = ladder(game=ttt(), budgets=PERFECT, num_rounds=2)

        records = rungs.tournament.run()

        assert records
        assert all(result.outcome == DRAW for _, _, result in records)

    def test_perfect_play_never_loses(self, ladder: LadderFactory, ttt: TTTFactory):
        game = ttt()
        rungs = ladder(game=game, budgets=PERFECT, num_rounds=1)
        weakest = RandomAI(game, seed=0)
        rungs.tournament.add_agent(weakest)

        records = rungs.tournament.run()
        _, info = rungs.tournament.get_by_description(weakest.description)

        winners = {
            pairing[0] if result.outcome == "A" else pairing[1]
            for _, pairing, result in records
            if result.outcome != DRAW
        }

        assert records
        assert info.id not in winners
