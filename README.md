# 12 ML Board Games

This repository contains a set of board games — (1) tic-tac-toe, (2) Connect Four and (3) chess — together with the agents that play them and the rating machinery that ranks those agents.

The long-term target is a chess AI trained with AlphaZero. The algorithms generalise across turn-taking, perfect-information, zero-sum games, so the simpler games are not detours so much as vital steps. Each game answers a specific question before we spend compute on the next.

| Stage | Game | Question it answers |
|---|---|---|
| 1 | Tic-tac-toe | Is the AlphaZero implementation correct at all, or is the code plain wrong? |
| 2 | Connect Four | Does it train on a non-trivial game, and how long does that take? |
| 3 | Chess | How far does this get on a single-GPU budget? |

## Games

**Tic-tac-toe** is solved. Rational players will always draw or win under perfect play. It exists here purely as a correctness test — a working AlphaZero implementation should reach perfect play almost immediately.

**Connect Four** is also solved: the first player wins with perfect play by opening in the centre. Crucially, solved means we have **ground truth**.

**Chess** is unsolved, and is the actual target.

## Agents

Three families:

**Random** — enumerates legal moves and picks one uniformly. The baseline that anchors the bottom of the ladder.

**Minimax** — minimax with alpha-beta pruning and iterative deepening, parametrised by a **node budget** per move. This gives a hierarchy: more nodes in general means stronger play.

Two notes on that parametrisation:

- We parametrise by node count, **not by fixed ply depth**. Search strength is not monotonic in depth — there is a well-known odd/even parity effect for many turn-taking games.
- Ladder quality depends strongly on the **leaf evaluation function**. Deeper search only helps if the eval is informative.

**AlphaZero** — policy/value network plus MCTS, trained purely through self-play. Detailed below.

## The Rating Ladder

Ratings use the standard logistic (Elo) model. The expected score — a number between 0 and 1 — of A playing against B is:

```python
def expected_score(r_a: Rating, r_b: Rating) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
```

An Elo difference of 400 means the stronger player has ten times the winning odds. The familiar incremental update

```python
r_next = r_prev + K * (game_outcome - expected_score)
```

where `game_outcome` is `1.0`, `0.5` or `0.0` for a win, draw or loss, is how Elo is usually understood, iteratively, and it is worth understanding — but it is **not** how we compute the ladder in one fell swoop.

### Playing conditions

Three requirements, all of which exist to stop us producing a ladder that is precise and wrong.

**Randomised tie-breaking.** Minimax with deterministic tie-breaking plays the identical move sequence every time it meets the same opponent. Fifty such games are fifty copies of one result, not fifty samples — but the rating fit treats them as independent and drives ratings apart with unearned confidence. Agents therefore break ties uniformly at random among equal-valued moves.

**Color reversal.** Every pairing is played in balanced colour pairs.

**Random opening book.** The first 2–4 plies are drawn from a fixed list of balanced openings, and each opening is played twice with colours reversed. The first moves are crucial and here the outcome is prone to high variance.

### Fitting ratings

For a fixed pool with a complete round-robin, the incremental update is the wrong tool. It is an *online* estimator, designed for a stream of games against a drifting population, similar to stochastic gradient descent, with the same limitations.

Instead we fit every rating simultaneously by maximum likelihood under the Bradley-Terry model. This is order-independent and admits confidence intervals:

```python
import numpy as np
from scipy.optimize import minimize

SCALE = 400 / np.log(10)  # converts Elo points to logits


def fit_ratings(n_players, games, prior_sd=200.0, anchor=1500.0):
    """games: iterable of (i, j, score_i), score_i in {0.0, 0.5, 1.0}"""
    i, j, s = (np.array(x) for x in zip(*games))

    def nll(r):
        d = (r[i] - r[j]) / SCALE
        ll = s * d - np.logaddexp(0.0, d)
        return -ll.sum() + (r**2).sum() / (2 * prior_sd**2)

    r = minimize(nll, np.zeros(n_players), method="L-BFGS-B").x
    return r - r.mean() + anchor
```

This is the same model the incremental formula implies.

**The prior is not optional.** Unregularised maximum likelihood can send rating differences to infinity. The prior keeps it finite.

**Confidence intervals** determine from bootstrapping.

### How many games to play?

For a single isolated match of `n` games between near-equal opponents, the standard error on the rating difference is:

$$\mathrm{SE}(d) = \frac{400}{\ln(10)\sqrt{p(1-p)\,n}} \approx \frac{347}{\sqrt{n}} \quad \text{at } p = 0.5$$

So an isolated 10-game match gives SE ≈ 110 Elo — a 95% interval of ±215, which is inadequate.

In a **full round-robin**, however, with all players playing all players, the fit pools information: each agent's rating is constrained by its games against every other agent simultaneously. In this case, with 20 players we get

| Games per pairing | Games per agent | SE | 95% CI |
|---|---|---|---|
| 10 | 190 | ±25 | ±49 |
| 20 | 380 | ±18 | ±35 |
| 50 | 950 | ±11 | ±22 |

20 players and 10 games is adequate for our purposes. Adding in color reversal we get a 95% confidence interval of ±35, which will be our target.

### The anchor pool

We pit 20 minimax agents — node budgets on a **log scale**, roughly 10 to 10,000 — plus the random agent against each other, fit the ratings, and **freeze them**. The pool mean is set to 1500.

> **1500 here means the average of this arbitrary pool of 20 minimax bots.** It bears no relationship to FIDE Elo or any human rating scale.

## Training AlphaZero

The loop is self-play → train → evaluate, repeated. These are **iterations**, not epochs — each one generates fresh data rather than passing over a fixed dataset.

```python
def train_and_evaluate(game: Game, model: Model, iterations: int) -> list[Rating]:
    history = []
    for i in range(iterations):
        games = self_play(game, model)
        train(model, games)
        rating = evaluate(game, model, ladder, checkpoints)

        history.append(rating)
        checkpoints.maybe_promote(model, rating, i)
    return history
```

### Evaluation

Evaluation holds the anchor ratings **fixed** and fits only the new agent's rating.

Two cost controls:

**Bracketing.** Only play the anchors nearest the current rating estimate.

**Checkpoint promotion.** The minimax ladder has a ceiling. Once the network beats the top anchor every time its rating becomes unmeasurable — at exactly the moment the project gets interesting. So past AlphaZero checkpoints are promoted into the anchor pool.

## Scope and compute

The bar here is beating a minimax agent at a few hundred nodes. In chess, with a branching factor around 35, a 200-node budget is roughly one to two ply — an agent that hangs pieces every few moves.
