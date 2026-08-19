import pandas as pd
import pytest

from win_prob_eval import real_state_at_quarter_start


def _pbp_row(**overrides):
    base = {
        "play_id": 1.0, "qtr": 1.0, "down": 1.0, "ydstogo": 10.0, "yardline_100": 75.0,
        "posteam": "KC", "defteam": "SF", "posteam_score": 0.0, "defteam_score": 0.0,
    }
    base.update(overrides)
    return base


def test_real_state_at_quarter_start_skips_kickoff_row_and_returns_first_scrimmage_play():
    game_df = pd.DataFrame([
        _pbp_row(play_id=1.0, qtr=2.0, down=float("nan"), posteam="SF", defteam="KC"),  # kickoff, no down
        _pbp_row(play_id=2.0, qtr=2.0, down=2.0, ydstogo=7.0, yardline_100=48.0,
                  posteam="KC", defteam="SF", posteam_score=7.0, defteam_score=3.0),
        _pbp_row(play_id=3.0, qtr=2.0, down=3.0, ydstogo=3.0),
    ])
    state = real_state_at_quarter_start(game_df, quarter=2)
    assert state is not None
    assert state.quarter == 2
    assert state.play_in_quarter == 0
    assert state.down == 2 and isinstance(state.down, int)
    assert state.ydstogo == 7 and isinstance(state.ydstogo, int)
    assert state.yardline_100 == 48 and isinstance(state.yardline_100, int)
    assert state.posteam == "KC" and state.defteam == "SF"
    assert state.posteam_score == 7 and state.defteam_score == 3


def test_real_state_at_quarter_start_returns_none_when_no_scrimmage_play_that_quarter():
    game_df = pd.DataFrame([
        _pbp_row(play_id=1.0, qtr=3.0, down=float("nan")),
    ])
    assert real_state_at_quarter_start(game_df, quarter=3) is None


def test_real_state_at_quarter_start_ignores_other_quarters():
    game_df = pd.DataFrame([
        _pbp_row(play_id=1.0, qtr=1.0, down=1.0),
        _pbp_row(play_id=2.0, qtr=3.0, down=1.0, ydstogo=10.0, yardline_100=60.0),
    ])
    state = real_state_at_quarter_start(game_df, quarter=3)
    assert state is not None
    assert state.quarter == 3
    assert state.yardline_100 == 60


def test_real_state_at_quarter_start_matches_real_2023_pbp_for_a_known_game():
    # Verifies against an actual real game (not just the constructed fixtures
    # above), per the design doc's testing section -- loads the real raw pbp
    # directly (not through win_prob_eval's own loader, to avoid depending on
    # a function this test file doesn't define until a later task) and checks
    # the function's output against an independently-computed expectation.
    pbp = pd.read_parquet("../data/pbp_2023.parquet")
    game_id = sorted(pbp["game_id"].unique())[0]
    game_df = pbp[pbp["game_id"] == game_id].sort_values("play_id")

    expected_rows = game_df[(game_df["qtr"] == 1) & game_df["down"].notna()]
    assert not expected_rows.empty, "test setup error: expected at least one real Q1 scrimmage play"
    expected = expected_rows.iloc[0]

    state = real_state_at_quarter_start(game_df, quarter=1)
    assert state is not None
    assert state.down == int(expected["down"])
    assert state.ydstogo == int(expected["ydstogo"])
    assert state.yardline_100 == int(expected["yardline_100"])
    assert state.posteam == expected["posteam"]
    assert state.defteam == expected["defteam"]
    assert state.posteam_score == int(expected["posteam_score"])
    assert state.defteam_score == int(expected["defteam_score"])
