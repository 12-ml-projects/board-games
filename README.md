# Board Games

Two board games — tic-tac-toe and Connect Four — the agents that play them, and the rating machinery that ranks those agents.

The target is an AlphaZero agent that learns Connect Four from self-play. The point of interest is not that it plays well but that its progress is **measured**: it climbs a ladder of opponents whose strengths were established first, so a training curve means something rather than being a plot of a loss that only goes down.

## Games

**Tic-tac-toe** is solved: perfect play always draws. It exists here as a correctness test, not a target. A search that never loses and always draws against itself is a strong signal that negamax, the terminal scoring and the evaluation convention are all right — and a bug in any of them shows up immediately.

**Connect Four** is also solved in principle (the first player wins by opening in the centre), but only by dedicated solvers. Nothing in this repository comes near perfect play, which is what makes it a usable target.

## Agents

**Random** picks uniformly among legal moves. It anchors the bottom of the ladder.

**Minimax** is negamax with alpha-beta pruning and iterative deepening, budgeted by **nodes searched per move** rather than by a fixed depth. One agent per budget gives a hierarchy of opponents from a single implementation.

**MCTS** is tree search with PUCT selection, budgeted by **simulations run per move**. It has no rollout: a leaf is judged by the same `evaluate` the minimax agents use, which is what makes the two comparable with the evaluation held fixed.

**AlphaZero** is the same tree search with a learned policy/value network supplying the priors and the leaf value at every move—essentially much like a human player trying to explore and evaluate what moves to consider.

The seam between AlphaZero and MCTS is a single protocol:

```python
class LeafEvaluator[StateT: State, MoveT: Move](Protocol):
    def __call__(
        self, game: Game[StateT, MoveT], state: StateT
    ) -> tuple[Mapping[MoveT, float], float]: ...
```

Priors over the legal moves, and the position's value from the side-to-move's perspective. The heuristic version returns flat priors and `game.evaluate(state)`; AlphaZero's returns both from a network.

Note that we use **PUCT in both**. If the two used different selection rules, comparing them would vary selection and evaluation at once and isolate neither.

Every agent implements one interface:

```python
class AI[StateT: State, MoveT: Move](ABC):
    @abstractmethod
    def choose_move(self, state: StateT) -> MoveT: ...
```

so anything that plays can be rated by the same machinery, and a new game plugs into all of it without the agents changing.

## The rating ladder

Ratings use the standard logistic (Elo) model. The expected score of A against B is

```python
def expected_score(r_a: Rating, r_b: Rating) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
```

A 400-point difference means ten times the winning odds.

### Fitting all ratings at once

The familiar incremental update — `r += K * (outcome - expected)` — is an *online* estimator built for a stream of games against a drifting population. It depends on the order games were played and never revisits early estimates. For a fixed pool played out in full, that is the wrong tool.

Instead every rating is fitted simultaneously by maximum likelihood under the **Bradley-Terry** model, using the MM iteration:

```
p_i  ←  W_i / Σ_j ( n_ij / (p_i + p_j) )
```

where `W_i` is agent *i*'s total score and `n_ij` the games between *i* and *j*. Iterate, renormalise, convert to Elo with `1500 + 400·log₁₀(p)`. This is order-independent, and it pools information: an agent's rating is constrained by every game it played, not only by its neighbours.

To avoid divergence of scores, we tune the choices of agents to make sure adjacent agents do not include a winner who wins 100% of the time.

### Playing conditions

**Colour reversal.** Every pairing is played from both sides in equal numbers. Bradley-Terry has no term for moving first, so its ratings are only trustworthy while that advantage is shared out evenly.

**Non-determinism.** Two deterministic agents replay one identical game, and fifty copies of a result are not fifty samples — but a rating fit treats them as independent and grows confident on evidence it does not have. Agents therefore break ties at random, and can optionally carry a noise parameter (below).

### How many games

Pairwise scores are reported with **Wilson** intervals rather than the usual normal approximation. The normal interval estimates its own width from the observed share, so at a clean sweep it reports zero width — twenty wins from twenty games comes out as exactly `[1, 1]`. Wilson asks instead which underlying shares could plausibly have produced what was seen, and stays honest at the extremes, which is where a ladder's most lopsided pairings live.

Sample size follows from the separation being measured, and the requirement is unintuitive: **games needed scales inversely with how decisive the pairing is.** The information a single game carries about a rating difference is proportional to `p(1−p)`, so

| true score | games for a ±0.05 interval |
|---|---|
| 0.55 | ~380 |
| 0.70 | ~320 |
| 0.90 | ~140 |

Fifty games can resolve a blowout and cannot resolve a near-tie. Since near-ties are also the *cheap* pairings, and blowouts the expensive ones, allocating games uniformly is the worst of both.

It also means a full round-robin wastes effort: a pairing at 0.99 tells you almost nothing per game. The final calibration run pairs each agent only with its nearest neighbours by rating, which spends the games where they buy something and keeps the comparison graph connected.

We ourselves choose to play adjacent agents, 2 on each side of the ladder.

## Experiments

Both experiments share one routine (`experiments/sweep.py`) and differ only in how they arrange the results. Each takes the agent family as a subcommand, since what counts as a budget differs — nodes searched for minimax, simulations run for MCTS — as does the setting worth varying alongside it:

```bash
python -m experiments.budget_grid.main minimax --nodes 400 --nodes 1600
python -m experiments.budget_grid.main mcts --simulations 50 --simulations 200
```

`experiments/variants.py` holds one entry per family, saying how to build an agent at a given budget and what to label the result. Nothing else in the experiments knows which family is running.

### Noise sensitivity

```bash
python -m experiments.noise_sensitivity.main minimax --help
python -m experiments.noise_sensitivity.main mcts --help
```

Agents that always pick the same move replay the same game, which makes a tournament uninformative. One fix is a noise parameter `ε`: each move available at the root has its search value lowered by its own draw from `[0, ε]`, and the best survivor is played. A separate draw per move — one shared amount would cancel out and change nothing.

Because nothing is lowered by more than `ε`, two moves further apart than `ε` keep their order. Only moves the agent already rates within `ε` of each other can swap, so `ε` reads as **the margin within which the agent stops caring**.

That also yields a guarantee. Evaluations are squashed into `±0.9` (`HEURISTIC_CEILING`) while a won position scores `±1.0`, so any `ε` below `0.1` leaves a gap noise cannot cross: a forced win is never passed over for a merely promising board, and a forced loss never mistaken for an ordinary bad move. `MiniMax` refuses to be constructed with an `ε` that breaks this.

**The experiment's conclusion was that noise is unnecessary here.** Across 0.0 to 0.099 the ladder looked the same, and at ε=0 the agents already produced 36 distinct games out of 60 — the random tie-breaking was supplying the variance by itself. Connect Four has at most seven legal moves and many exact ties, so this is a fact about this game; a game with a wider branching factor would need the experiment re-run.

#### Searching the root on a full window

Adding noise forced a change to the search itself.

Alpha-beta prunes by proving a move *cannot be the best one*, and then stopping. It never works out how much worse that move actually is. Normally that costs nothing, because only the best move is ever played — but what it reports for a rejected move is an upper bound, "no better than this", rather than a value.

Noise compares values, so those bounds stop being harmless. A move that loses on the spot might be cut off early and reported as "no better than 0.3", leaving it just under the leader where a nudge of `ε` is enough to pick it. The noise would be guarding a number already wrong by more than a point.

For MCTS the swept setting is the **exploration constant** rather than noise — MCTS has no noise parameter, and its own source of non-determinism is which branches the search happens to favour.

### Budget grid

```bash
python -m experiments.budget_grid.main minimax --help
python -m experiments.budget_grid.main mcts --help
```

Every budget against every other, one score matrix per setting.

**The experiment's outcome was that the ladder is noisy below 400 and saturates above 25600 nodes** if we employ a factor of 4. This gives us a small ladder of 4 agents. A random agent makes this 5 agents to compare against and a ladder to climb.

### Evaluation

Evaluation holds the anchor ratings **fixed** and fits only the new agent's.

**Evaluate against the negamax agents.** Evaluate against adjacent agents to get an estimated Elo, starting with the weakest agent. The next time an evaluation tournament is held, place the agent on the estimated Elo.

**Checkpoint promotion.** If the agent improves past the ladder, it can still be played against itself at an earlier point in time and evaluated as such.

## A note on AI assistance

This repository was written with substantial help from an AI assistant (Claude), used to help guide implementation, refactoring, profiling and running the experiments. The design decisions are mine: what to measure, how to parametrise the ladder, which results to trust, writing example tests, test fixtures, what to throw away. The boilerplate, overall architecture (ABCs, modules etc.) and algorithms are my own, from extensive reading.

AI assistance has been used and reviewed to fill in test suites, find mistakes, writing documentation, fill in gaps, boilerplate, discuss ideas, and run numerous throw-away experiments.

Where the assistant was most useful was as a fast way to settle questions rather than discuss them — writing a throwaway script, running it, and reporting what came back. Several of the findings above came out that way, including some that contradicted what either of us expected.