"""
Bucketing for situational play fields. Boundaries chosen from the real
distributions we inspected on the 2022-2023 seasons (not guessed).
"""

import numpy as np

PLAY_TYPES = [
    "pass", "run", "no_play_penalty", "no_play_timeout", "no_play_other",
    "punt", "field_goal", "kickoff", "extra_point", "qb_kneel", "qb_spike",
]
PLAY_TYPE_TO_IDX = {p: i for i, p in enumerate(PLAY_TYPES)}

DOWN_BUCKETS = ["1", "2", "3", "4", "none"]  # none = no-down plays (kickoff, PAT, etc.)

# ydstogo is heavily spiked at 0 (no-down plays) and 10 (1st-and-10).
DISTANCE_BUCKETS = ["none", "1-3", "4-6", "7-9", "10", "11+"]

# yardline_100: 0 = opponent goal line, 100 = own goal line
FIELD_ZONE_BUCKETS = ["own_deep", "own_territory", "midfield", "opp_territory", "red_zone"]

# score_differential (posteam - defteam) is concentrated near zero (IQR -7..+4),
# so buckets are tighter near zero and wider at the tails, not linear.
# NOTE: these are UPPER BOUNDS (inclusive) per bucket, one fewer than the number
# of labels -- the last label catches everything above the last bound. See
# bucket_edges() for the exact semantics, and its docstring for why this
# matters (an earlier version of this file had an off-by-one that put
# boundary values in the wrong bucket).
SCORE_DIFF_EDGES = [-17, -9, -3, 2, 8, 16]
SCORE_DIFF_BUCKETS = ["down_17+", "down_9-16", "down_3-8", "close", "up_3-8", "up_9-16", "up_17+"]

QUARTER_BUCKETS = ["1", "2", "3", "4", "OT"]

TIME_BUCKETS = [">10min", "5-10min", "2-5min", "<2min"]

YARDS_GAINED_EDGES = [-1, 0, 3, 6, 10, 20]
YARDS_GAINED_BUCKETS = ["loss", "no_gain", "short_1-3", "medium_4-6", "good_7-10", "big_11-20", "explosive_20+"]

RETURN_YARDS_EDGES = [0, 5, 15, 30]
RETURN_YARDS_BUCKETS = ["none", "short_1-5", "medium_6-15", "long_16-30", "explosive_30+"]


def bucket_play_type(row):
    pt = row["play_type"]
    if pt == "no_play":
        sub = row.get("play_type_nfl")
        if sub == "PENALTY":
            return "no_play_penalty"
        if sub == "TIMEOUT":
            return "no_play_timeout"
        return "no_play_other"
    if pt in PLAY_TYPE_TO_IDX:
        return pt
    return "no_play_other"  # NaN / anything unrecognized -- caller should have already dropped these rows


def bucket_down(down):
    if down in (1.0, 2.0, 3.0, 4.0):
        return str(int(down))
    return "none"


def bucket_distance(ydstogo, down_bucket):
    if down_bucket == "none":
        return "none"
    y = ydstogo
    if y <= 3:
        return "1-3"
    if y <= 6:
        return "4-6"
    if y <= 9:
        return "7-9"
    if y == 10:
        return "10"
    return "11+"


def bucket_field_zone(yardline_100):
    if yardline_100 is None or np.isnan(yardline_100):
        return "midfield"  # rare missing case, e.g. some kickoffs
    y = yardline_100
    if y >= 80:
        return "own_deep"
    if y >= 50:
        return "own_territory"
    if y >= 40:
        return "midfield"
    if y >= 20:
        return "opp_territory"
    return "red_zone"


def bucket_score_diff(diff):
    return bucket_edges(diff, SCORE_DIFF_EDGES, SCORE_DIFF_BUCKETS)


def bucket_quarter(qtr):
    if qtr in (1.0, 2.0, 3.0, 4.0):
        return str(int(qtr))
    return "OT"


def bucket_time(game_seconds_remaining, qtr):
    # seconds remaining *in the current quarter*, not the game, is what actually
    # matters for two-minute-warning-style situational behavior.
    if qtr in (1.0, 2.0, 3.0, 4.0):
        secs_in_period = game_seconds_remaining - (4 - int(qtr)) * 900 if qtr <= 4 else game_seconds_remaining
    else:
        secs_in_period = game_seconds_remaining
    if secs_in_period is None or np.isnan(secs_in_period):
        secs_in_period = 900
    if secs_in_period > 600:
        return ">10min"
    if secs_in_period > 300:
        return "5-10min"
    if secs_in_period > 120:
        return "2-5min"
    return "<2min"


def bucket_edges(value, edges, labels):
    """
    `edges` are inclusive UPPER bounds, one fewer than `labels`:
    value <= edges[0]            -> labels[0]
    edges[i-1] < value <= edges[i] -> labels[i]
    value > edges[-1]            -> labels[-1]
    e.g. edges=[-1,0,3], labels=[loss,no_gain,short_1-3,big] means
    -1 and below -> loss; exactly 0 -> no_gain; 1,2,3 -> short_1-3; 4+ -> big.
    (side='left' is what gives this exact inclusive-upper-bound behavior --
    side='right' silently shifts every boundary value into the next bucket.)
    """
    if value is None or np.isnan(value):
        value = 0
    idx = np.searchsorted(edges, value, side="left")
    idx = min(idx, len(labels) - 1)
    return labels[idx]
