from tests.integration.conftest import TTTFactory


class TestTTT:
    def test_initial_state(self, ttt: TTTFactory):
        tic_tac_toe = ttt()

        assert tic_tac_toe.initial().board == (("", "", ""), ("", "", ""), ("", "", ""))
