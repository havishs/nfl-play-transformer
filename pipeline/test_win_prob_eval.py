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


from win_prob_eval import real_outcome


def _game_row(**overrides):
    base = {"home_team": "KC", "away_team": "SF", "home_score": 24.0, "away_score": 20.0}
    base.update(overrides)
    return base


def test_real_outcome_team_a_is_home_and_wins():
    game_df = pd.DataFrame([_game_row(home_score=24.0, away_score=20.0)])
    assert real_outcome(game_df, team_a="KC") == 1.0


def test_real_outcome_team_a_is_away_and_loses():
    game_df = pd.DataFrame([_game_row(home_score=24.0, away_score=20.0)])
    assert real_outcome(game_df, team_a="SF") == 0.0


def test_real_outcome_tie_returns_half():
    game_df = pd.DataFrame([_game_row(home_score=17.0, away_score=17.0)])
    assert real_outcome(game_df, team_a="KC") == 0.5


from types import SimpleNamespace

import torch

from win_prob_eval import rollout_outcome, win_probability, state_seed
from generate import GameState


def _final_state(**overrides):
    base = dict(quarter=4, play_in_quarter=10, down=1, ydstogo=10, yardline_100=50,
                posteam="KC", defteam="SF", posteam_score=17, defteam_score=10)
    base.update(overrides)
    return GameState(**base)


def test_rollout_outcome_team_a_wins_when_team_a_has_the_ball_and_leads():
    sim = SimpleNamespace(team_a="KC", team_b="SF")
    log = [{"event": "gain", "state": _final_state(posteam="KC", defteam="SF", posteam_score=17, defteam_score=10)}]
    assert rollout_outcome(sim, log, log[0]["state"]) == "team_a"


def test_rollout_outcome_team_b_wins_when_team_a_is_on_defense_and_trails():
    sim = SimpleNamespace(team_a="KC", team_b="SF")
    # team_a (KC) is on defense here; defteam_score (KC's score) is lower than posteam_score (SF's)
    log = [{"event": "gain", "state": _final_state(posteam="SF", defteam="KC", posteam_score=24, defteam_score=10)}]
    assert rollout_outcome(sim, log, log[0]["state"]) == "team_b"


def test_rollout_outcome_unresolved_when_still_tied_in_overtime():
    sim = SimpleNamespace(team_a="KC", team_b="SF")
    log = [{"event": "gain", "state": _final_state(quarter=6, posteam_score=20, defteam_score=20)}]
    assert rollout_outcome(sim, log, log[0]["state"]) == "unresolved"


def test_rollout_outcome_falls_back_to_initial_state_when_log_is_empty():
    sim = SimpleNamespace(team_a="KC", team_b="SF")
    initial = _final_state(posteam="KC", defteam="SF", posteam_score=3, defteam_score=0)
    assert rollout_outcome(sim, [], initial) == "team_a"


def test_win_probability_tallies_wins_and_unresolved_as_half_credit(monkeypatch):
    # fake sim whose .generate() cycles through team_a win, team_b win, unresolved
    outcomes = [
        _final_state(posteam="KC", defteam="SF", posteam_score=10, defteam_score=0),   # team_a win
        _final_state(posteam="SF", defteam="KC", posteam_score=10, defteam_score=0),   # team_b win
        _final_state(quarter=6, posteam_score=10, defteam_score=10),                    # unresolved
    ]
    calls = {"i": 0}

    class FakeSim:
        team_a = "KC"
        team_b = "SF"

        def generate(self, n_plays, initial_state, generator=None):
            state = outcomes[calls["i"] % len(outcomes)]
            calls["i"] += 1
            return [{"event": "gain", "state": state}]

    initial = _final_state()
    p_hat = win_probability(FakeSim(), initial, n_rollouts=3, base_seed=0, max_plays=10)
    # 1 team_a win (1.0) + 1 team_b win (0.0) + 1 unresolved (0.5) = 1.5 / 3
    assert p_hat == pytest.approx(0.5)


def test_state_seed_is_deterministic_and_varies_by_game_and_quarter():
    a = state_seed("2023_01_KC_SF", 1)
    b = state_seed("2023_01_KC_SF", 1)
    c = state_seed("2023_01_KC_SF", 2)
    d = state_seed("2023_02_KC_SF", 1)
    assert a == b
    assert a != c
    assert a != d


from win_prob_eval import sample_validation_games


@pytest.fixture(scope="module")
def small_dataset_for_sampling():
    from dataset import PlayDataset
    return PlayDataset(history_seasons=[2022], training_seasons=[2023], data_dir="../data", max_examples=1500)


def test_sample_validation_games_is_deterministic(small_dataset_for_sampling):
    first = sample_validation_games(small_dataset_for_sampling, n_games=3, seed=42)
    second = sample_validation_games(small_dataset_for_sampling, n_games=3, seed=42)
    assert first == second


def test_sample_validation_games_stays_within_the_held_out_split(small_dataset_for_sampling):
    from get_batch import build_game_index
    from train import VAL_FRACTION

    game_ids = sorted(build_game_index(small_dataset_for_sampling.examples).keys())
    n_val = max(1, round(len(game_ids) * VAL_FRACTION))
    val_game_ids = set(game_ids[-n_val:])

    sampled = sample_validation_games(small_dataset_for_sampling, n_games=3, seed=42)
    assert set(sampled) <= val_game_ids


def test_sample_validation_games_caps_at_available_val_games(small_dataset_for_sampling):
    sampled = sample_validation_games(small_dataset_for_sampling, n_games=10_000, seed=1)
    from get_batch import build_game_index
    assert len(sampled) <= len(build_game_index(small_dataset_for_sampling.examples))
