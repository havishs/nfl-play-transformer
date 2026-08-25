# NFL Play-by-Play Transformer

A decoder-only transformer that generates NFL games autoregressively, one play at a time, and a Monte Carlo harness that turns those generated rollouts into a win-probability estimate — evaluated against real outcomes, not just per-play accuracy.

## What this is

Most NFL prediction projects treat a game as a fixed feature vector — final score, box-score stats — and regress on it. This project instead models a game as a sequence: ~174 real plays, each with situational context (down/distance/field position/score) and full 22-man personnel (11 offense + 11 defense), fed through a GPT-style causal decoder. Predicting one play at a time and rolling the model forward autoregressively means the model has to actually learn drive-level football logic (a 3rd-and-2 behaves differently than a 3rd-and-15) rather than just memorizing aggregate stats.

The architecture is a four-piece stack:

1. **Player history encoder** — per-player causal rolling-average stats (career-to-date, position-aware). Currently hand-computed rather than learned.
2. **In-game team form** — a live, causally-updated EMA of each team's in-game offensive/defensive performance, evolving as the model generates.
3. **Nested play-level attention** — the 11 offensive and 11 defensive players on a given play attend to each other, producing one play-summary token. This replaces hard-coded matchup assignment (e.g. "which CB covers which WR"), which isn't in any public NFL dataset — the attention layer learns implicit matchups instead.
4. **Outer causal decoder** — a standard GPT-style decoder over the play-summary tokens, causal across an entire game.

Full architecture rationale, data-source citations, and design tradeoffs: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Key results

### Per-play prediction

| Target | Result | Baseline |
|---|---|---|
| Touchdown | Recall 0.9% → 15.9% after fixing a loss-weighting bug (below) | — |
| Turnover | No measurable improvement from any fix tried | — |
| Yards gained (7-class bucket) | 36% accuracy | 23.7% (majority class) |
| Return yards | ~70% accuracy | ~65% (majority class) |

**Touchdown prediction was fixed, not just tuned.** It was pinned exactly to the majority-class baseline through multiple training runs — more data, more steps, a real player-history encoder, none of it moved the needle. Diagnosed as gradient starvation: the model's loss was an unweighted sum across four output heads, and touchdown's skewed distribution (~3.7% positive rate) gave it a naturally tiny gradient relative to `yards_gained`. Fixed with entropy-derived per-head loss reweighting. Recall went 0.9% → 15.9% (~18x) with precision improving too — not a precision/recall tradeoff, genuinely more signal extracted.

**Turnover prediction is the honest negative result.** Two independent, differently-mechanismed fixes were tried — the same loss reweighting that fixed touchdown, and a separate in-game team-form feature addressing missing live context — and turnover moved **zero** in both: 0 positive predictions ever, Brier score statistically identical to the trivial base-rate baseline to 4 decimal places. Since touchdown responded to the loss fix and turnover didn't despite equal treatment, this isn't a training-dynamics problem — it's that none of the features tried contain exploitable pre-snap signal for a turnover on a specific play. The likely explanation: turnovers are substantially driven by in-play randomness (a tipped pass, a bad snap) that isn't observable before the snap, unlike touchdown likelihood, which correlates strongly with field position — something directly visible pre-snap.

### Monte Carlo win-probability evaluation

The real test of an autoregressive game model isn't per-play accuracy — it's whether rolling it forward many times from a given game state produces a usable win probability. The harness (`pipeline/win_prob_eval.py`) takes a real held-out game at a real point in the game (start of Q1/Q2/Q3/Q4), runs 500 seeded Monte Carlo rollouts forward through the model, and tallies how often each team wins — then compares that probability against what actually happened.

Evaluated on 40 held-out games, run as two independent, **guaranteed non-overlapping** 20-game samples (`results/win_prob_run1.csv`, `results/win_prob_run2_fresh_games.csv`) to confirm the pattern wasn't a fluke of one sample:

| Quarter | Accuracy | 95% CI | Brier |
|---|---|---|---|
| Q1 (pregame-equivalent) | 45.0% | [30.7%, 60.2%] | 0.291 |
| Q2 | 52.5% | — | 0.275 |
| Q3 | 65.0% | — | 0.190 |
| Q4 (late in-game) | **80.0%** | **[65.2%, 89.5%]** | **0.130** |

**The model has no reliable pregame signal** — Q1 accuracy is statistically indistinguishable from a coin flip (its 95% CI comfortably contains 50%), and is worse than even the simple home-team-always-wins baseline (~57%). An early version of this result (on a single 20-game sample) showed Q1 accuracy *below* a coin flip, which looked like it might be exploitable by just betting against the model — testing that hypothesis on a second, disjoint sample of games showed it was sampling noise, not a real pattern: Q1 came back at exactly 50% on the fresh set.

**But the model's estimate sharpens sharply and reliably as it sees more of the game** — accuracy climbs monotonically (45% → 52.5% → 65% → 80%) and Brier score more than halves (0.291 → 0.130), both replicated independently across the two disjoint samples. By Q4, the 95% CI (**[65%, 90%]**) no longer contains chance, and the point estimate lands at the top of the live in-game win-probability baseline range cited in the original project scoping (75-80%).

Reproducibility was verified in production, not just in unit tests: every rollout's random draws — including special-teams outcomes (punt distance, field goal make/miss) that previously used an unseeded global RNG — are threaded through a single seeded generator per game-state. Two completely separate runs of the same 5 games produced bit-for-bit identical win-probability outputs.

## Repo structure

```
pipeline/               all source + tests (131 tests, pytest)
  build_dataset.py      assembles training examples from raw play-by-play
  model.py               GameTransformer: nested attention + causal decoder
  train.py, eval.py     training loop, per-play precision/recall/Brier eval
  generate.py            GameSimulator: one autoregressive game rollout
  win_prob_eval.py       Monte Carlo win-probability harness
  team_form.py, special_teams_features.py, player_features.py, situational.py
                          feature engineering modules, each single-purpose
  test_*.py              one test file per module above
results/                 raw per-state CSVs from the win-probability evaluation
PROJECT_BRIEF.md         full architecture doc, data-source citations,
                          known limitations, immediate next steps
colab_setup.ipynb        end-to-end Colab notebook (fetch data → train →
                          evaluate win probability), GPU training
```

## Running it

**Training and evaluation happen on Colab** (`colab_setup.ipynb`) — clones this repo, fetches six seasons of play-by-play via `nfl_data_py`, trains on GPU, restores/persists checkpoints and results to Google Drive between sessions, and runs the Monte Carlo win-probability harness. See the notebook for the exact one-time setup (a GitHub token in Colab's Secrets manager).

**Tests run locally:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install nfl_data_py==0.3.3 --no-deps  # see requirements.txt's install note
cd pipeline && pytest .
```

## Known limitations (accepted, not bugs)

- **No offensive line data.** No public per-play OL grades exist anywhere in nflverse (that's PFF-proprietary); OL player vectors fall back to bio/usage features only.
- **No coverage-assignment data.** Nothing in public NFL data says "this CB covered that WR" — the nested attention layer exists specifically to learn implicit matchups without it.
- **Personnel is fixed for a whole rollout.** No substitution modeling, no in-game roster changes.
- **Kickoffs are always a touchback.** No returns modeled; blocked kicks and 2-point conversions aren't modeled either.
- **Turnover prediction**, per the results above, appears to be near the information-theoretic limit of what's predictable from the available pre-snap features.

Full list, with the real-data verification behind each one: [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md).

## Future work

- Replace the hand-computed player-history encoder (piece 1 of the architecture) with a real learned encoder, now that the harder architectural risk (nested attention inside a causal decoder) is proven to train.
- Per-player in-game attribution — the one untested lever for turnover prediction, requires the model to also predict *who* touched the ball on a generated play.
- A later, separate phase (not started): profile the trained model and hand-write a CUDA kernel for its bottleneck op.
