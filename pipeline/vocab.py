"""
Index mappings for every categorical field a play example can contain.

Situational fields (play_type, down, distance, ...) use the bucket lists
already defined in situational.py -- those are fixed regardless of which
seasons are loaded. Team and position codes are NOT hardcoded: they're built
from whatever seasons are actually loaded, since team codes shift over time
(franchise relocations) and this keeps the vocab honest about what the data
contains. Both include an "UNK" entry as a fallback for codes not seen at
vocab-build time (e.g. a team/position present in a later season but not in
the seasons used to build this vocab).
"""

import pandas as pd

import situational as sit


def _to_vocab(labels):
    return {label: i for i, label in enumerate(labels)}


PLAY_TYPE_VOCAB = _to_vocab(sit.PLAY_TYPES)
DOWN_VOCAB = _to_vocab(sit.DOWN_BUCKETS)
DISTANCE_VOCAB = _to_vocab(sit.DISTANCE_BUCKETS)
FIELD_ZONE_VOCAB = _to_vocab(sit.FIELD_ZONE_BUCKETS)
SCORE_DIFF_VOCAB = _to_vocab(sit.SCORE_DIFF_BUCKETS)
QUARTER_VOCAB = _to_vocab(sit.QUARTER_BUCKETS)
TIME_VOCAB = _to_vocab(sit.TIME_BUCKETS)
YARDS_GAINED_VOCAB = _to_vocab(sit.YARDS_GAINED_BUCKETS)
RETURN_YARDS_VOCAB = _to_vocab(sit.RETURN_YARDS_BUCKETS)


def build_team_vocab(seasons, data_dir="../data"):
    teams = set()
    for s in seasons:
        pbp = pd.read_parquet(f"{data_dir}/pbp_{s}.parquet", columns=["posteam", "defteam"])
        teams |= set(pbp["posteam"].dropna()) | set(pbp["defteam"].dropna())
    return _to_vocab(sorted(teams) + ["UNK"])


def build_position_vocab(seasons, data_dir="../data"):
    positions = set()
    for s in seasons:
        roster = pd.read_parquet(f"{data_dir}/roster_{s}.parquet", columns=["position"])
        positions |= set(roster["position"].dropna())
    return _to_vocab(sorted(positions) + ["UNK"])


def build_vocabs(seasons, data_dir="../data"):
    """All vocabs needed to tensorize a play example, keyed by field name."""
    return {
        "play_type": PLAY_TYPE_VOCAB,
        "down": DOWN_VOCAB,
        "distance": DISTANCE_VOCAB,
        "field_zone": FIELD_ZONE_VOCAB,
        "score_diff": SCORE_DIFF_VOCAB,
        "quarter": QUARTER_VOCAB,
        "time_bucket": TIME_VOCAB,
        "yards_gained": YARDS_GAINED_VOCAB,
        "return_yards": RETURN_YARDS_VOCAB,
        "team": build_team_vocab(seasons, data_dir),
        "position": build_position_vocab(seasons, data_dir),
    }
