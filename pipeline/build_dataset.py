"""
Assembles one training example per play:
  - situational fields (bucketed)
  - 22 personnel (11 offense + 11 defense), each with position + causal feature vector
  - outcome targets for THIS play (yards/touchdown/turnover/return), which
    combined with the situational fields deterministically produce the next
    play's starting situation during autoregressive generation.

Meta rows (play_type NaN -- END_QUARTER, GAME_START, etc.) are dropped: they
aren't real plays and carry no personnel/situational signal.
"""

import numpy as np
import pandas as pd

import situational as sit
import team_form as tf
from player_features import PlayerFeatureLookup


def load_pbp(seasons, data_dir="../data"):
    frames = [pd.read_parquet(f"{data_dir}/pbp_{s}.parquet") for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df[df["play_type"].notna()].copy()  # drop meta rows
    df = df[df["offense_players"].notna() & df["defense_players"].notna()].copy()  # need personnel
    print(f"  kept {len(df)}/{before} rows after dropping meta rows and missing-personnel rows")
    return df.sort_values(["game_id", "play_id"]).reset_index(drop=True)


def build_situational(row):
    down_b = sit.bucket_down(row["down"])
    return {
        "play_type": sit.bucket_play_type(row),
        "down": down_b,
        "distance": sit.bucket_distance(row["ydstogo"], down_b),
        "field_zone": sit.bucket_field_zone(row["yardline_100"]),
        "score_diff": sit.bucket_score_diff(row["score_differential"]),
        "quarter": sit.bucket_quarter(row["qtr"]),
        "time_bucket": sit.bucket_time(row["game_seconds_remaining"], row["qtr"]),
        "posteam": row["posteam"],
        "defteam": row["defteam"],
    }


SCRIMMAGE_YARDAGE_TYPES = {"pass", "run", "qb_kneel", "qb_spike"}   # "yards gained" is a real rushing/passing concept
TD_TURNOVER_APPLICABLE_TYPES = {"pass", "run", "punt", "kickoff"}    # td/turnover can happen on these
RETURN_TYPES = {"punt", "kickoff"}                                    # always has a return concept


def build_targets(row):
    play_type = row["play_type"]
    is_turnover = bool(row.get("interception") == 1 or row.get("fumble_lost") == 1)

    yards_applicable = play_type in SCRIMMAGE_YARDAGE_TYPES
    td_turnover_applicable = play_type in TD_TURNOVER_APPLICABLE_TYPES
    # return_yards is meaningful on punts/kickoffs always, or on a pass/run that ended in a turnover
    return_applicable = (play_type in RETURN_TYPES) or (play_type in ("pass", "run") and is_turnover)

    return {
        "yards_gained": sit.bucket_edges(row["yards_gained"], sit.YARDS_GAINED_EDGES, sit.YARDS_GAINED_BUCKETS)
                         if yards_applicable else None,
        "yards_gained_applicable": yards_applicable,
        "touchdown": bool(row.get("touchdown") == 1) if td_turnover_applicable else None,
        "turnover": is_turnover if td_turnover_applicable else None,
        "td_turnover_applicable": td_turnover_applicable,
        "return_yards": sit.bucket_edges(row.get("return_yards", 0), sit.RETURN_YARDS_EDGES, sit.RETURN_YARDS_BUCKETS)
                         if return_applicable else None,
        "return_yards_applicable": return_applicable,
    }


def build_personnel(row, lookup):
    season, week = int(row["season"]), int(row["week"])
    off_ids = row["offense_players"].split(";")
    def_ids = row["defense_players"].split(";")
    offense = [lookup.lookup(pid, season, week) for pid in off_ids]
    defense = [lookup.lookup(pid, season, week) for pid in def_ids]
    return offense, defense


def build_examples(history_seasons, training_seasons, data_dir="../data", max_examples=None):
    print("loading player feature lookup (history seasons:", history_seasons, ")")
    lookup = PlayerFeatureLookup(sorted(set(history_seasons) | set(training_seasons)), data_dir)

    print("loading pbp for training seasons:", training_seasons)
    pbp = load_pbp(training_seasons, data_dir)

    examples = []
    for game_id, game in pbp.groupby("game_id"):
        game = game.sort_values("play_id")
        form_state = tf.initial_team_form()
        for _, row in game.iterrows():
            offense, defense = build_personnel(row, lookup)
            targets = build_targets(row)
            examples.append({
                "game_id": game_id,
                "play_id": row["play_id"],
                "week": int(row["week"]),
                "situational": build_situational(row),
                "targets": targets,
                "offense": offense,
                "defense": defense,
                "team_form": tf.team_form_features(form_state, row["posteam"], row["defteam"]),
            })
            form_state = tf.update_team_form(
                form_state, row["posteam"], row["defteam"], row["yards_gained"],
                targets["yards_gained_applicable"], targets["touchdown"], targets["turnover"],
                targets["td_turnover_applicable"],
            )
            if max_examples and len(examples) >= max_examples:
                return examples
    return examples


if __name__ == "__main__":
    examples = build_examples(history_seasons=[2022], training_seasons=[2023], max_examples=20)
    print(f"\nbuilt {len(examples)} example plays\n")

    ex = examples[10]
    print("=== EXAMPLE PLAY ===")
    print("game:", ex["game_id"], "| play_id:", ex["play_id"], "| week:", ex["week"])
    print("situational:", ex["situational"])
    print("targets:", ex["targets"])
    print()
    print("offense[0] (first personnel slot):", ex["offense"][0])
    print()
    print("defense[0] (first personnel slot):", ex["defense"][0])
