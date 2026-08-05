# NFL Play-by-Play Transformer — Project Brief

## Read this first
This is a handoff from a design/exploration session with Claude (chat) to
Claude Code. The person building this already worked through Karpathy's
"Let's build GPT" tutorial end to end (bigram -> embeddings -> self-attention
-> multi-head attention -> feedforward -> residual -> LayerNorm -> full
decoder-only transformer on char-level Shakespeare) and understands attention
mechanics, causal masking, and the training loop cold. **Do not re-explain
attention from zero.** Do walk through genuinely new concepts (nested
attention, multi-field embeddings, the encoder/decoder split below) the same
incremental, example-driven way Karpathy's tutorial does.

Working style that matters: this person wants to **understand and iterate on
the design**, not receive a finished repo. Build incrementally, verify
against real data before committing to a design (this whole project has been
built that way -- every field/bucket/architecture choice below was checked
against actual nfl_data_py output, not assumed), and explain *why* before
writing code that depends on it. When you hit a genuinely ambiguous design
fork, surface it rather than silently picking one.

---

## What we're building

Predict NFL game outcomes (score, TDs, passing yards, etc.) not as separate
targets, but by modeling a full game as a sequence of plays and training a
GPT-style decoder-only transformer to autoregressively generate the game,
play by play. Final stats fall out of simulating the rollout -- the same way
a language model doesn't have a separate "predict the theme" head.

**Why play-level, not game-level:** a game-level model (one token per past
game) throws away the situational detail that determines the score, and with
only ~17 games/team/season there isn't enough sequence data anyway. Play-level
tokens fix both: ~174 plays/game (measured, not estimated) x ~270 games/season
x multiple seasons is a real dataset.

---

## Architecture: the four-piece stack (current design, as of this handoff)

The original plan was a flat Karpathy-style decoder operating on team-level
situational fields only (down/distance/field position/score/etc., no player
identity -- that was pushed to a "v2"). **That plan was superseded during this
session** in favor of a richer design once we confirmed the data could
actually support it. Current design:

1. **Player History Encoder** -- reads a player's pre-game career history
   (bidirectional/non-causal is fine here, since it's summarizing something
   already finished) and outputs a "prior ability" embedding per player.
   **STUBBED FOR NOW**: instead of a learned encoder, use hand-computed causal
   rolling-average stats (already implemented, see `pipeline/player_features.py`).
   Upgrade to a real learned encoder only after step 3+4 below are proven to
   train.

2. **In-game running form state** -- a player's form *within the current
   game*, updated live as the decoder generates forward. Different timescale
   and different causality rules than (1): career history is fixed and known,
   in-game form updates as we generate. **STUBBED FOR NOW** too -- not built
   yet at all. Start with a simple EMA over recent per-play outcome features
   once (1) and (2)/(3)/(4) below are wired up; no need for a learned
   recurrent cell yet.

3. **Nested play-level attention** -- within a single play, the 11 offensive
   and 11 defensive personnel on the field attend to each other (offense-11 x
   defense-11), producing one play-summary token. This replaces any attempt
   to hard-code "which DB covers which WR" -- **that data doesn't exist**
   publicly (see Data Findings below), so implicit matchup learning via
   attention is the only real option, and it's a good one.

4. **Outer causal decoder** -- the existing Karpathy backbone (Head,
   MultiHeadAttention, FeedForward, Block, causal mask, training loop,
   generate loop) operating over the play-summary tokens produced by (3),
   one token per play, causal across time within a single game.

**Why cross-attention is back in the design** (the original doc explicitly
ruled it out, reasoning "no separate source sequence to encode"): a player's
career history *is* a separate, non-causal source sequence now. That's a
textbook encoder-decoder split -- (1) is the encoder, (4) is the
decoder, and the decoder cross-attends into per-player encoder outputs when
that player is on the field for the current play.

**Staging rationale (why stub 1+2 first):** (3)+(4) carry the real
architectural risk -- does nested attention inside a causal sequence
transformer even train sensibly. (1)+(2) are comparatively well-understood
feature engineering once the backbone is proven. De-risk the expensive
unknown first with cheap stand-ins for the parts we're confident we can build
later.

**Known real limitation, not solved, don't try to over-engineer around it:**
OL (offensive line) has no public per-play performance data anywhere in
nflverse -- no pass-block/run-block grades (that's PFF-proprietary). OL player
vectors fall back to bio/usage features only (years experience, games
started, draft capital, snap count). Expect this to be the noisiest part of
the player vector. This is an accepted, documented gap, not a bug to chase.

**Player quality vector design:** the person wants "sum of the vector ~=
overall quality, but the vector is multi-faceted." This does NOT happen for
free from a plain `nn.Embedding` -- it must be engineered. Agreed approach:
**real-stat initialization** -- seed each player's embedding from actual
aggregate stats (position-appropriate; see Data Findings), then let training
fine-tune from there instead of from random noise. (Two other options were
discussed and rejected/deferred: a hierarchical scalar-plus-style-vector
split, and an auxiliary regression loss tying a linear readout to a real
metric -- worth revisiting if real-stat init alone doesn't give the desired
property.)

---

## Data findings (verified against real nfl_data_py output, not assumed)

### Install gotcha
`pip install nfl_data_py` fails on a fresh environment: it pins
`pandas<2.0,>=1.0`, and building pandas<2.0 from source on Python 3.12 fails
with `ModuleNotFoundError: No module named 'pkg_resources'` (recent
setuptools removed it). **Fix:** `pip install nfl_data_py --no-deps`, then
separately `pip install pandas numpy requests appdirs pyarrow` (modern
versions). Confirmed working at runtime with pandas 3.0.2 / numpy 2.4.4
despite the stale pin -- it's a metadata problem, not a real compatibility
issue.

### Scale
2023 season: 49,665 total rows, 397 columns. After dropping meta rows
(`play_type` NaN -- these are `GAME_START`/`END_QUARTER`/`END_GAME` markers,
not real plays, ~1,452 rows) and rows missing personnel data (~3%): **46,160
real plays**. 285 games, ~174 plays/game (min 139, max 218) -- higher than
the doc's original ~150 estimate.

### Full 22-man personnel IS available
`offense_players` and `defense_players` columns: semicolon-delimited gsis IDs,
11 per side, present on 93% of plays. This is what makes the player-level
design possible at all.

### What's NOT available: coverage assignment
No column or dataset anywhere in public nflverse data says "this DB covered
that WR." We know *who was on the field*, never *who was matched against
whom*. This is why nested attention (design point 3 above) exists -- it's the
substitute for hard-coded 1:1 assignment. Checked `import_ftn_data` (FTN
charting: personnel counts, motion, play-action, blitz counts, etc.) too --
good situational charting, still no coverage assignment.

### Returns/turnovers are already baked into the same play row
Checked a real interception: `yards_gained=0`, `return_yards=1`, and the very
next play's `yardline_100` already reflects the post-return spot. Same for
punts. **No separate "return" token/row is needed** -- one play's row fully
determines the next play's starting position once you have `yards_gained`,
`return_yards`, and the touchdown/turnover flags.

### Per-player stat sources, by position group
| Position group | Source | Key fields |
|---|---|---|
| QB/RB/WR/TE | `nfl.import_weekly_data(years)` | `passing_epa`, `rushing_epa`, `receiving_epa`, `target_share` (per-week, so causal rolling averages are possible) |
| DB/LB/DL | `nfl.import_weekly_pfr('def', years)` | `def_completion_pct`, `def_passer_rating_allowed`, `def_pressures`, `def_sacks`, `def_missed_tackle_pct` (also per-week) |
| OL | *(none)* | bio-only fallback: `years_exp`, `draft_number` from `import_seasonal_rosters` |
| K/P | `import_weekly_data` / seasonal | FG%, punt avg -- available, low priority so far |

Season-level PFR stats (`import_seasonal_pfr`) exist too but are NOT
causally safe to use for in-season rolling features -- they aggregate the
whole season including future games relative to any given play. Use the
**weekly** versions only for anything that will be used as a causal feature.

### ID crosswalk needed
`weekly_pfr` keys on `pfr_player_id`; `offense_players`/`defense_players` use
gsis `player_id`. `import_seasonal_rosters` has both (`player_id`, `pfr_id`)
-- join through that. Coverage of the crosswalk on the full roster is only
58%, but 91% for players who actually recorded defensive stats (inactive/
practice-squad players without stats don't matter for this).

### Cross-season carryover is required, not optional
Tested: a Week 1 2023 play showed **every single player** (including a
5-year-veteran starter) as "no prior data," because only 2023 was loaded.
Verified the causal join logic itself is correct by checking a Week 10 play
(correctly averaged 8-9 prior in-season games, correctly zero leakage from
future weeks). **Fix implemented:** `PlayerFeatureLookup` takes a
`history_seasons` argument separate from `training_seasons` -- pull at least
one prior season purely for lookback, don't draw training examples from it.
This problem doesn't fully go away at any finite history window (Week 1 of
the *earliest* season in scope will always be relatively cold) -- accepted,
not solved.

### Schedule endpoint is blocked / unnecessary
`nfl.import_schedules()` fetches from `http://www.habitatring.com/games.csv`,
which returned `403 Forbidden` in this sandboxed environment (likely a
network allowlist issue, may or may not reproduce on your machine). Turned
out to be unnecessary anyway -- `season` and `week` are already columns
directly on the play-by-play data, sufficient for chronological ordering.

---

## Two real bugs found and fixed this session (watch for this pattern)

Both are the same underlying failure mode: **something that isn't a
meaningful value gets silently treated as a legitimate one.** Worth internalizing
as a category, since it's likely to recur elsewhere in this pipeline
(e.g. when the personnel lists have fewer than 11 players some weeks, when a
game is missing weekly stats for a bye week, etc.) -- check for it deliberately anywhere
new numeric defaulting gets introduced.

1. **Off-by-one in bucket boundaries.** `np.searchsorted(edges, value,
   side='right') - 1` put boundary values in the wrong bucket (e.g. a
   score_diff of exactly -17 landed in `down_9-16` instead of `down_17+`).
   Fixed by switching to `side='left'` with edges-as-inclusive-upper-bounds
   semantics (see docstring in `pipeline/situational.py::bucket_edges`).
   **A 42-case boundary test suite now exists** (`pipeline/test_situational.py`)
   and passes -- run it after touching any bucketing logic.

2. **"Not applicable" vs. "true zero."** `yards_gained` is a literal `0.0`
   (not NaN) in the source data for `extra_point`/`field_goal`/`kickoff`/
   `no_play` rows, because "yards gained" as a rushing/passing concept simply
   doesn't apply to those play types -- but naively bucketing it produced an
   artificially inflated 39% "no_gain" rate. Fixed with explicit
   per-play-type applicability flags (`yards_gained_applicable`,
   `td_turnover_applicable`, `return_yards_applicable` in
   `build_dataset.py::build_targets`) -- target value is `None` when not
   applicable, and the eventual training loop must mask these out of the
   loss rather than training on a fake zero.

---

## What's already built and validated (in `pipeline/`)

- `situational.py` -- bucketing for all situational fields (play_type, down,
  distance, field_zone, score_diff, quarter, time_bucket, yards_gained,
  return_yards), all boundaries chosen from real data distributions, not
  guessed. Includes the corrected `bucket_edges` helper.
- `test_situational.py` -- 42-case boundary regression suite, all passing.
- `player_features.py` -- `PlayerFeatureLookup` class: builds causal
  cumulative-mean stat tables per player across seasons, with proper
  temporal cutoffs (a game at `(season, week)` only ever sees strictly
  earlier `(season, week)` rows), position-aware feature sourcing (offense
  vs. defense vs. bio-only OL fallback), and `has_off_stats`/`has_def_stats`
  flags to distinguish real cold-start from position-inapplicability.
- `build_dataset.py` -- assembles one example per play: bucketed situational
  fields + 22 personnel (each with position + causal feature vector) +
  applicability-masked outcome targets. Drops meta rows and
  personnel-missing rows. Tested end-to-end on 2023 (training) with 2022 as
  history-only lookback.

**Currently cached in `data/`** (bundled with this handoff, ~42MB total):
`pbp_2022.parquet`, `pbp_2023.parquet`, `weekly_off_2022/2023.parquet`,
`weekly_def_2022/2023.parquet` (already gsis-ID-mapped), `roster_2022/2023.parquet`.

**Explicitly not done yet:**
- Only 2022 (history) + 2023 (training) -- not the full 2016-2023 range from
  the original doc scope. Deliberate: prove the architecture trains on a
  small slice before spending time/bandwidth pulling 6 more seasons.
- Output is still Python dicts, not tensors. No vocab dictionaries built for
  categorical fields yet. No PyTorch `Dataset`/`DataLoader` yet.
- No model code written yet at all (backbone, nested attention block, or
  output heads).

---

## Immediate next steps (in order)

1. **Vocab dictionaries + tensorization.** Every categorical field in
   `situational.py` (play_type, down, distance, field_zone, score_diff,
   quarter, time_bucket, team codes, position) needs an index mapping. Build
   a `PyTorch Dataset` that wraps `build_dataset.py`'s output and produces
   batched tensors: situational fields as indices, personnel as
   `(B, T, 22, feature_dim)` tensors, targets as indices + applicability
   masks.
2. **`get_batch`**, adapted from Karpathy's version: sample a random game,
   then a random window of consecutive plays *within that game* -- do not
   let a window span two games (this was already flagged as a requirement in
   the original design doc and still holds).
3. **Model v0**: nested attention block (offense-11 x defense-11 -> one
   play-summary vector) feeding the outer causal decoder (reused Karpathy
   backbone) with stubbed history/form features (i.e. steps 1-2 of the
   four-piece stack, hand-computed, not learned). Output heads:
   `play_type`, `yards_gained` (masked by applicability), `touchdown`
   (masked), `turnover` (masked), `return_yards` (masked). Loss = masked sum
   across heads.
4. Confirm it trains (loss decreases, no NaN blowups) and that `generate()`
   produces plausible rollouts before doing anything else.
5. Only after 4 works: replace the stub Player History Encoder with a real
   learned encoder (step 1 of the stack), then build the in-game running form
   state (step 2, start with EMA).
6. Scale to the full 2016-2023 season range.
7. Evaluate against baselines (below) -- calibration/Brier score, not just
   accuracy.

---

## Baselines to beat (established via research, not guessed)

- Home team always wins: ~57% (dumbest possible baseline)
- Vegas closing line, straight-up: ~66-68% -- the real ceiling, most public
  models don't beat this
- Published academic ML (logistic regression/RF/XGBoost) on pregame stats:
  60s-low 70s
- Against the spread (ATS): ~52-55% is normal; much higher should be treated
  with suspicion (spread is engineered near a coin flip)
- **In-game live win probability models: ~75-80%** -- this is the most
  relevant comparison for this project, since partway through a simulated
  rollout the model is functionally doing in-game win probability
- Be skeptical of any single study claiming 85%+ pregame straight-up accuracy
  on a small dataset -- classic overfitting/small-test-set red flag

---

## Open questions, not yet resolved

- Exact bucket boundaries were chosen from real 2023 distributions but not
  re-validated across the full multi-season range -- may need adjustment once
  more seasons are loaded.
- Loss weighting across output heads (currently would be an unweighted sum --
  may need tuning if one head dominates gradient).
- `block_size` (play-history window length) tuning -- not yet tested at all.
- Whether special-teams-specific outcomes (FG make/miss + distance, punt
  fair-catch/touchback, kickoff touchback) deserve their own dedicated output
  heads rather than being squeezed into the same schema as scrimmage plays --
  flagged during the applicability-masking fix, not resolved.
- v3 idea from the original doc (cross-attention between two teams' full
  histories, "how has this team performed against opponents like their
  upcoming one") -- may now be partially subsumed by the Player History
  Encoder once that's real, but worth revisiting explicitly once that's built.

---

## Original project doc

The original scoping doc (data source: `nfl_data_py`/nflfastR, build order,
starting field table, etc.) that kicked off this whole project is available
on request from the person if you need the very first framing -- most of its
content is superseded or incorporated above, but it also contains reasoning
about why play-level beats game-level modeling that's still fully valid and
wasn't repeated in full here.
