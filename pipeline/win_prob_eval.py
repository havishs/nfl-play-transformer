"""
Monte Carlo win-probability evaluation harness.

For a sample of held-out validation games, at the real start of each
quarter, runs many seeded rollouts through GameSimulator and tallies a win
probability, then compares it against the real final score. See
docs/superpowers/specs/2026-08-19-monte-carlo-win-probability-design.md for
the full design and the compute-budget reasoning behind the constants below.
"""

import hashlib

import torch

from generate import GameState

QUARTERS = [1, 2, 3, 4]
N_EVAL_GAMES = 25
ROLLOUTS_PER_STATE = 500
MAX_PLAYS = 400
RESULTS_CSV_PATH = "win_prob_results.csv"


def real_state_at_quarter_start(game_df, quarter):
    """
    First real scrimmage play (down not null) in `quarter`, for one game's
    raw pbp rows. `game_df` must already be sorted by play_id. Returns a
    GameState with play_in_quarter=0 (this harness's own definition of
    "start of quarter"), or None if the quarter has no scrimmage play at all
    (e.g. a game that ran out the clock without a real 4th-quarter snap).
    """
    rows = game_df[(game_df["qtr"] == quarter) & game_df["down"].notna()]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return GameState(
        quarter=quarter, play_in_quarter=0,
        down=int(row["down"]), ydstogo=int(row["ydstogo"]),
        yardline_100=int(row["yardline_100"]),
        posteam=row["posteam"], defteam=row["defteam"],
        posteam_score=int(row["posteam_score"]), defteam_score=int(row["defteam_score"]),
    )


def real_outcome(game_df, team_a):
    """1.0 if team_a actually won this real game, 0.0 if it lost, 0.5 on a real tie."""
    row = game_df.iloc[0]
    team_a_score = row["home_score"] if team_a == row["home_team"] else row["away_score"]
    team_b_score = row["away_score"] if team_a == row["home_team"] else row["home_score"]
    if team_a_score > team_b_score:
        return 1.0
    if team_a_score < team_b_score:
        return 0.0
    return 0.5


def rollout_outcome(sim, log, initial_state):
    """
    "team_a", "team_b", or "unresolved" (MAX_PLAYS exhausted mid-overtime,
    still tied -- counted as 0.5/0.5 credit by win_probability).
    """
    final = log[-1]["state"] if log else initial_state
    if final.quarter > 4 and final.posteam_score == final.defteam_score:
        return "unresolved"
    if final.posteam == sim.team_a:
        team_a_score, team_b_score = final.posteam_score, final.defteam_score
    else:
        team_a_score, team_b_score = final.defteam_score, final.posteam_score
    return "team_a" if team_a_score > team_b_score else "team_b"


def state_seed(game_id, quarter):
    """
    Stable (process-independent) integer seed for one (game, quarter) starting
    point, used as the base seed for that state's rollouts (rollout i uses
    base_seed + i). Uses hashlib rather than Python's built-in hash(), which
    is randomized per-process and would break reproducibility across runs.
    """
    digest = hashlib.sha256(f"{game_id}:{quarter}".encode()).hexdigest()
    return int(digest[:8], 16)


def win_probability(sim, initial_state, n_rollouts, base_seed, max_plays):
    wins_a = 0.0
    for i in range(n_rollouts):
        generator = torch.Generator().manual_seed(base_seed + i)
        log = sim.generate(max_plays, initial_state, generator=generator)
        outcome = rollout_outcome(sim, log, initial_state)
        if outcome == "team_a":
            wins_a += 1.0
        elif outcome == "unresolved":
            wins_a += 0.5
    return wins_a / n_rollouts
