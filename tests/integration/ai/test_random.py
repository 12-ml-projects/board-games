import pytest

from src.exceptions import NoLegalMovesException
from src.tic_tac_toe.state import TTTState
from tests.integration.ai.conftest import RandomAIFactory
from tests.integration.conftest import TTTFactory


class TestRandomAI:
    def test_choose_move(self, random_ai: RandomAIFactory, ttt: TTTFactory):
        game = ttt()
        initial_state = game.initial()
        ai = random_ai(game)
        move = ai.choose_move(initial_state)

        assert move is not None

    def test_choose_move_throws_if_no_legal_moves(
        self, random_ai: RandomAIFactory, ttt: TTTFactory
    ):
        game = ttt()
        final_state = TTTState(
            player="A", board=(("x", "o", "x"), ("x", "o", "x"), ("o", "x", "o"))
        )
        ai = random_ai(game)

        with pytest.raises(NoLegalMovesException, match="No legal moves available"):
            ai.choose_move(final_state)
