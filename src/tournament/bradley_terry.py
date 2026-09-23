from collections import defaultdict
from math import log10
from typing import Final, Sequence
from uuid import UUID

from src.game.outcome import DRAW, ONGOING, Outcome
from src.tournament.elo import BASELINE_ELO
from src.tournament.match import MatchRecord

# Strengths only mean anything next to each other, so each sweep is rescaled
# to put the average agent here.
AVERAGE_STRENGTH: Final[float] = 1.0


def bradley_terry(
    records: Sequence[MatchRecord],
    *,
    prior_games: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-12,
) -> dict[UUID, float]:
    relevant_records = [r for r in records if r[2].outcome != ONGOING]
    agent_ids = {r[1][0] for r in relevant_records} | {
        r[1][1] for r in relevant_records
    }

    wins = get_wins(relevant_records, agent_ids)

    p = {agent_id: AVERAGE_STRENGTH for agent_id in agent_ids}
    for _ in range(max_iterations):
        new_p = normalise(get_new_p(p, wins, prior_games))
        settled = max(abs(new_p[agent_id] - p[agent_id]) for agent_id in agent_ids)
        p = new_p

        if settled < tolerance:
            break

    return {agent_id: BASELINE_ELO + 400.0 * log10(s) for agent_id, s in p.items()}


def get_wins(
    records: Sequence[MatchRecord], agent_ids: set[UUID]
) -> dict[UUID, dict[UUID, float | None]]:
    win_matrix: dict[UUID, dict[UUID, float | None]] = defaultdict(dict)

    for agent_id in agent_ids:
        for opponent_id in agent_ids:
            if agent_id == opponent_id:
                win_matrix[agent_id][opponent_id] = None
                continue

            # Which slot an agent sat in says nothing about it, so match on
            # the pairing regardless of order.
            wins = sum(
                points_won_by(agent_id, pairing, record.outcome)
                for _, pairing, record in records
                if {agent_id, opponent_id} == set(pairing)
            )

            win_matrix[agent_id][opponent_id] = wins

    return win_matrix


def points_won_by(
    agent_id: UUID, pairing: tuple[UUID, UUID], outcome: Outcome
) -> float:
    """What `agent_id` took from one match: 1 for a win, 0.5 for a draw."""
    if outcome == DRAW:
        return 0.5

    id_a, id_b = pairing
    winner = id_a if outcome == "A" else id_b

    return 1.0 if winner == agent_id else 0.0


def get_new_p(
    prev_p: dict[UUID, float],
    wins: dict[UUID, dict[UUID, float | None]],
    prior_games: float,
) -> dict[UUID, float]:
    p = defaultdict(lambda: 1.0)

    for own_id in wins.keys():
        # The prior is a phantom opponent of average strength that everyone
        # has already drawn with to avoid divergence common in MLE methods.
        won = prior_games / 2 + sum(
            [
                win * prev_p[other_id] / (prev_p[other_id] + prev_p[own_id])
                for other_id, win in wins[own_id].items()
                if other_id != own_id and win is not None
            ]
        )
        lost = prior_games / 2 / (prev_p[own_id] + AVERAGE_STRENGTH) + sum(
            [
                (wins[other_id].get(own_id) or 0.0)
                / (prev_p[other_id] + prev_p[own_id])
                for other_id in wins.keys()
                if other_id != own_id
            ]
        )

        p[own_id] = won / lost

    return p


def normalise(p: dict[UUID, float]) -> dict[UUID, float]:
    mean = sum(log10(strength) for strength in p.values()) / len(p)
    scale = 10**mean / AVERAGE_STRENGTH

    return {agent_id: strength / scale for agent_id, strength in p.items()}
