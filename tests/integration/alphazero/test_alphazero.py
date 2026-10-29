from math import isclose

import torch

from src.ai.random import RandomAI
from src.alphazero import AlphaZero, NetEvaluator, build_net, encode
from src.connect_four.move import C4Move
from src.tournament.match import Match
from tests.integration.conftest import C4Factory


class TestEncode:
    def test_shape(self, c4: C4Factory):
        game = c4()

        assert encode(game.initial()).shape == (3, 6, 7)

    def test_planes_follow_the_side_to_move(self, c4: C4Factory):
        game = c4()
        after_a = game.make_move(game.initial(), C4Move(column=3))
        after_b = game.make_move(after_a, C4Move(column=3))

        # B to move: A's stone at the bottom of column 3 is "theirs"
        planes = encode(after_a)
        assert planes[0].sum() == 0
        assert planes[1][0][3] == 1
        assert planes[2].sum() == 0

        # A to move: its own stone is now "mine", B's is "theirs"
        planes = encode(after_b)
        assert planes[0][0][3] == 1
        assert planes[1][1][3] == 1
        assert planes[:2].sum() == 2
        assert bool((planes[2] == 1).all())


class TestNetEvaluator:
    def test_priors_cover_legal_moves_and_sum_to_one(self, c4: C4Factory):
        torch.manual_seed(0)
        game = c4()
        evaluate = NetEvaluator(build_net(game))

        priors, value = evaluate(game, game.initial())

        assert set(priors) == set(game.legal_moves(game.initial()))
        assert isclose(sum(priors.values()), 1.0, rel_tol=1e-5)
        assert -1.0 <= value <= 1.0

    def test_full_columns_get_no_prior(self, c4: C4Factory):
        torch.manual_seed(0)
        game = c4()
        state = game.initial()
        # Alternating stones never connect vertically
        for _ in range(6):
            state = game.make_move(state, C4Move(column=0))

        priors, _ = NetEvaluator(build_net(game))(game, state)

        assert C4Move(column=0) not in priors
        assert isclose(sum(priors.values()), 1.0, rel_tol=1e-5)


class TestAlphaZero:
    def test_plays_a_full_game(self, c4: C4Factory):
        torch.manual_seed(0)
        game = c4()
        agent = AlphaZero(game, build_net(game), n_simulations=16, seed=0)

        result = Match(agent, RandomAI(game, seed=0), game).play()

        assert game.is_final_state(result.states[-1])

    def test_description(self, c4: C4Factory):
        game = c4()
        agent = AlphaZero(game, build_net(game), n_simulations=16)

        assert agent.description == "alphazero(simulations=16)"
