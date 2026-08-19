"""
Monte Carlo win-probability evaluation harness.

For a sample of held-out validation games, at the real start of each
quarter, runs many seeded rollouts through GameSimulator and tallies a win
probability, then compares it against the real final score. See
docs/superpowers/specs/2026-08-19-monte-carlo-win-probability-design.md for
the full design and the compute-budget reasoning behind the constants below.
"""

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
