from collections import Counter
from itertools import combinations

from src.game.outcome import ONGOING
from tests.integration.conftest import TTTFactory
from tests.integration.tournament.conftest import MinimaxTournamentFactory

HYPERPARAMS = [(5, 0), (25, 0), (125, 0)]
DESCRIPTIONS = [f"minimax(max_nodes={nodes})" for nodes, _ in HYPERPARAMS]


class TestTournament:
    def test_ladder_order(
        self, minimax_tournament: MinimaxTournamentFactory, ttt: TTTFactory
    ):
        tournament = minimax_tournament(
            game=ttt(), hyperparams=HYPERPARAMS, num_rounds=10
        )
        tournament.run()

        _, agent_info_5 = tournament.get_by_description("minimax(max_nodes=5)")
        _, agent_info_25 = tournament.get_by_description("minimax(max_nodes=25)")
        _, agent_info_125 = tournament.get_by_description("minimax(max_nodes=125)")

        assert agent_info_5.elo < agent_info_25.elo
        assert agent_info_25.elo < agent_info_125.elo

    def test_tournament_is_reproducible(
        self, minimax_tournament: MinimaxTournamentFactory, ttt: TTTFactory
    ):
        first, second = (self._run_and_rate(minimax_tournament, ttt) for _ in range(2))

        assert first == second

    def test_every_pairing_is_played_from_both_sides(
        self, minimax_tournament: MinimaxTournamentFactory, ttt: TTTFactory
    ):
        tournament = minimax_tournament(
            game=ttt(), hyperparams=HYPERPARAMS, num_rounds=4
        )
        records = tournament.run()

        games_as_alpha = Counter(pairing[0] for _, pairing, _ in records)
        games_played = Counter(
            agent_id for _, pairing, _ in records for agent_id in pairing
        )

        assert games_as_alpha.keys() == games_played.keys()
        assert all(
            games_as_alpha[agent_id] * 2 == played
            for agent_id, played in games_played.items()
        )

    def test_plays_every_pairing_each_round(
        self, minimax_tournament: MinimaxTournamentFactory, ttt: TTTFactory
    ):
        num_rounds = 4
        tournament = minimax_tournament(
            game=ttt(), hyperparams=HYPERPARAMS, num_rounds=num_rounds
        )

        records = tournament.run()

        pairings = len(list(combinations(HYPERPARAMS, 2)))
        assert len(records) == num_rounds * pairings * 2

    def test_every_match_is_played_out(
        self, minimax_tournament: MinimaxTournamentFactory, ttt: TTTFactory
    ):
        tournament = minimax_tournament(
            game=ttt(), hyperparams=HYPERPARAMS, num_rounds=4
        )

        records = tournament.run()

        assert records
        assert all(result.outcome != ONGOING for _, _, result in records)

    def _run_and_rate(
        self, minimax_tournament: MinimaxTournamentFactory, ttt: TTTFactory
    ) -> dict[str, float]:
        tournament = minimax_tournament(
            game=ttt(), hyperparams=HYPERPARAMS, num_rounds=6
        )
        tournament.run()

        return {
            agent.description: info.elo
            for agent, info in map(tournament.get_by_description, DESCRIPTIONS)
        }
