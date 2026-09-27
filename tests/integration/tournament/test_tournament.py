from collections import Counter
from itertools import combinations

from src.game.game import Game
from src.game.outcome import ONGOING
from tests.integration.tournament.conftest import BUDGETS, LadderFactory


class TestTournament:
    def test_ladder_order(self, ladder: LadderFactory, game: Game):
        rungs = ladder(game=game, budgets=BUDGETS, num_rounds=10)
        rungs.tournament.run()

        elos = rungs.elos()

        assert elos == sorted(elos)

    def test_tournament_is_reproducible(self, ladder: LadderFactory, game: Game):
        # Agent UUIDs are freshly minted per tournament, so the records of two
        # identical runs never compare equal. What has to match is the ratings.
        first, second = (self._run_and_rate(ladder, game) for _ in range(2))

        assert first == second

    def test_every_pairing_is_played_from_both_sides(
        self, ladder: LadderFactory, game: Game
    ):
        # Bradley-Terry has no term for moving first, so its ratings are only
        # meaningful while going first is shared out evenly.
        records = ladder(game=game, budgets=BUDGETS, num_rounds=3).tournament.run()

        games_as_alpha = Counter(pairing[0] for _, pairing, _ in records)
        games_played = Counter(
            agent_id for _, pairing, _ in records for agent_id in pairing
        )

        assert games_as_alpha.keys() == games_played.keys()
        assert all(
            games_as_alpha[agent_id] * 2 == played
            for agent_id, played in games_played.items()
        )

    def test_plays_every_pairing_each_round(self, ladder: LadderFactory, game: Game):
        num_rounds = 3
        rungs = ladder(game=game, budgets=BUDGETS, num_rounds=num_rounds)

        records = rungs.tournament.run()

        pairings = len(list(combinations(rungs.descriptions, 2)))
        assert len(records) == num_rounds * pairings * 2

    def test_every_match_is_played_out(self, ladder: LadderFactory, game: Game):
        records = ladder(game=game, budgets=BUDGETS, num_rounds=3).tournament.run()

        assert records
        assert all(result.outcome != ONGOING for _, _, result in records)

    def _run_and_rate(self, ladder: LadderFactory, game: Game) -> dict[str, float]:
        rungs = ladder(game=game, budgets=BUDGETS, num_rounds=6)
        rungs.tournament.run()

        return dict(zip(rungs.descriptions, rungs.elos()))
