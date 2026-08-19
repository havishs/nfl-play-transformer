from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import team_form as tf
from dataset import PlayDataset
from generate import GameSimulator, GameState
from get_batch import build_game_index
from model import GameTransformer
from train import VAL_FRACTION
from win_prob_eval import (
    evaluate,
    game_rows,
    load_pbp_for_seasons,
    real_form_state_before,
    real_outcome,
    real_state_at_quarter_start,
    rollout_outcome,
    sample_validation_games,
    state_seed,
    summarize,
    win_probability,
)


@pytest.fixture(scope="module")
def small_dataset():
    return PlayDataset(history_seasons=[2022], training_seasons=[2023], data_dir="../data", max_examples=1500)


@pytest.fixture(scope="module")
def simulator(small_dataset):
    torch.manual_seed(0)
    model = GameTransformer(small_dataset.vocabs, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0)
    model.eval()
    seed_game_id = small_dataset.examples[0]["game_id"]
    return GameSimulator(model, small_dataset, seed_game_id, device="cpu")


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


def test_real_form_state_before_returns_empty_for_q1():
    # Q1 legitimately starts empty -- no real plays precede it.
    game_df = pd.DataFrame([
        _pbp_row(play_id=1.0, qtr=1.0, down=1.0, play_type="run", posteam="KC", defteam="SF",
                 yards_gained=5.0, touchdown=0.0, interception=0.0, fumble_lost=0.0),
    ])
    assert real_form_state_before(game_df, quarter=1) == tf.initial_team_form()


def test_real_form_state_before_replays_real_prior_plays_for_a_later_quarter():
    game_df = pd.DataFrame([
        _pbp_row(play_id=1.0, qtr=1.0, down=1.0, play_type="run", posteam="KC", defteam="SF",
                 yards_gained=5.0, touchdown=0.0, interception=0.0, fumble_lost=0.0),
        # the Q2 kickoff row itself (down is null) should also be replayed --
        # it precedes the cutoff (Q2's first real scrimmage play) by play_id
        _pbp_row(play_id=2.0, qtr=2.0, down=float("nan"), play_type="kickoff", posteam="SF", defteam="KC",
                 yards_gained=0.0, touchdown=0.0, interception=0.0, fumble_lost=0.0),
        _pbp_row(play_id=3.0, qtr=2.0, down=1.0, play_type="pass", posteam="KC", defteam="SF",
                 yards_gained=3.0, touchdown=0.0, interception=0.0, fumble_lost=0.0),
    ])
    form_state = real_form_state_before(game_df, quarter=2)
    # Only row 1 and row 2 (play_id < 3, the Q2 cutoff) should be reflected --
    # row 3 (the Q2 starting point itself) must NOT be replayed.
    assert form_state != tf.initial_team_form()
    assert form_state["KC"]["offense"]["has_yards_history"] is True
    assert form_state["KC"]["offense"]["yards_ema"] == pytest.approx(5.0)
    assert form_state["SF"]["defense"]["has_yards_history"] is True
    assert form_state["SF"]["defense"]["yards_ema"] == pytest.approx(5.0)


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
        form_state = tf.initial_team_form()

        def generate(self, n_plays, initial_state, generator=None):
            state = outcomes[calls["i"] % len(outcomes)]
            calls["i"] += 1
            return [{"event": "gain", "state": state}]

    initial = _final_state()
    p_hat = win_probability(FakeSim(), initial, n_rollouts=3, base_seed=0, max_plays=10,
                             initial_form_state=tf.initial_team_form())
    # 1 team_a win (1.0) + 1 team_b win (0.0) + 1 unresolved (0.5) = 1.5 / 3
    assert p_hat == pytest.approx(0.5)


def test_win_probability_final_form_state_matches_a_single_fresh_rollout(simulator):
    # Regression test for form_state contamination across "independent"
    # rollouts: GameSimulator.generate() mutates sim.form_state as a side
    # effect and never resets it, so without an explicit reset each
    # successive rollout within one win_probability() call would see a
    # team_form input polluted by every prior rollout's fictitious plays.
    # If form_state is correctly reset before each rollout, then running
    # win_probability with n_rollouts=3 should leave sim.form_state in
    # exactly the same state as running a single rollout with the seed of
    # the *last* of those three (base_seed + n_rollouts - 1) -- because each
    # rollout starts from a clean slate regardless of what came before it in
    # the loop.
    initial_state = GameState(
        quarter=1, play_in_quarter=0, down=1, ydstogo=10, yardline_100=75,
        posteam=simulator.team_a, defteam=simulator.team_b, posteam_score=0, defteam_score=0,
    )
    initial_form_state = tf.initial_team_form()

    win_probability(simulator, initial_state, n_rollouts=3, base_seed=0, max_plays=10,
                     initial_form_state=initial_form_state)
    form_state_after_three = simulator.form_state

    simulator.form_state = initial_form_state
    generator = torch.Generator().manual_seed(0 + 2)  # base_seed + (n_rollouts - 1)
    simulator.generate(10, initial_state, generator=generator)
    form_state_after_one_matching_seed = simulator.form_state

    assert form_state_after_three == form_state_after_one_matching_seed


def test_win_probability_does_not_mutate_the_initial_form_state_it_is_given(simulator):
    # win_probability reassigns the SAME initial_form_state dict object to
    # sim.form_state at the top of every rollout, rather than a copy. That's
    # only safe because team_form.update_team_form() always returns a new
    # dict instead of mutating its input -- confirm the object handed in is
    # untouched after a full win_probability() call, and that calling it
    # twice with the same initial_form_state produces the same result both
    # times (proving neither call left contamination behind for the other).
    initial_state = GameState(
        quarter=1, play_in_quarter=0, down=1, ydstogo=10, yardline_100=75,
        posteam=simulator.team_a, defteam=simulator.team_b, posteam_score=0, defteam_score=0,
    )
    initial_form_state = tf.initial_team_form()
    snapshot_before = dict(initial_form_state)

    p_hat_first = win_probability(simulator, initial_state, n_rollouts=3, base_seed=0, max_plays=10,
                                   initial_form_state=initial_form_state)
    assert initial_form_state == snapshot_before

    p_hat_second = win_probability(simulator, initial_state, n_rollouts=3, base_seed=0, max_plays=10,
                                    initial_form_state=initial_form_state)
    assert initial_form_state == snapshot_before
    assert p_hat_first == p_hat_second


def test_state_seed_is_deterministic_and_varies_by_game_quarter_and_seed():
    a = state_seed(1, "2023_01_KC_SF", 1)
    b = state_seed(1, "2023_01_KC_SF", 1)
    c = state_seed(1, "2023_01_KC_SF", 2)
    d = state_seed(1, "2023_02_KC_SF", 1)
    e = state_seed(2, "2023_01_KC_SF", 1)
    assert a == b
    assert a != c
    assert a != d
    assert a != e


def test_sample_validation_games_is_deterministic(small_dataset):
    first = sample_validation_games(small_dataset, n_games=3, seed=42)
    second = sample_validation_games(small_dataset, n_games=3, seed=42)
    assert first == second


def test_sample_validation_games_stays_within_the_held_out_split(small_dataset):
    game_ids = sorted(build_game_index(small_dataset.examples).keys())
    n_val = max(1, round(len(game_ids) * VAL_FRACTION))
    val_game_ids = set(game_ids[-n_val:])

    sampled = sample_validation_games(small_dataset, n_games=3, seed=42)
    assert set(sampled) <= val_game_ids


def test_sample_validation_games_caps_at_available_val_games(small_dataset):
    sampled = sample_validation_games(small_dataset, n_games=10_000, seed=1)
    assert len(sampled) <= len(build_game_index(small_dataset.examples))


def test_summarize_hand_computed_accuracy_and_brier():
    records_by_quarter = {
        1: [(0.8, 1.0), (0.3, 0.0)],  # both correct
        2: [(0.5, 1.0)],               # p_hat == 0.5 always counts as incorrect
    }
    summary = summarize(records_by_quarter)

    assert summary[1]["n"] == 2
    assert summary[1]["accuracy"] == pytest.approx(1.0)
    assert summary[1]["brier"] == pytest.approx(((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2)

    assert summary[2]["n"] == 1
    assert summary[2]["accuracy"] == pytest.approx(0.0)
    assert summary[2]["brier"] == pytest.approx((0.5 - 1.0) ** 2)

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["accuracy"] == pytest.approx(2 / 3)


def test_summarize_handles_an_empty_quarter():
    summary = summarize({1: [], 2: [(0.9, 1.0)]})
    assert summary[1]["n"] == 0
    assert summary[1]["accuracy"] != summary[1]["accuracy"]  # NaN
    assert summary["overall"]["n"] == 1


def test_evaluate_end_to_end_smoke(small_dataset):
    torch.manual_seed(0)
    model = GameTransformer(small_dataset.vocabs, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0)
    model.eval()

    game_id = small_dataset.examples[0]["game_id"]
    season = int(game_id.split("_")[0])
    pbp = load_pbp_for_seasons([season], data_dir="../data")

    records_by_quarter, csv_rows, skipped = evaluate(
        model, small_dataset, pbp, [game_id], device="cpu",
        rollouts_per_state=3, max_plays=200,
    )

    total_points = sum(len(v) for v in records_by_quarter.values())
    assert total_points + skipped == 4  # one point per quarter, for the one game evaluated
    assert len(csv_rows) == total_points
    for records in records_by_quarter.values():
        for p_hat, y in records:
            assert 0.0 <= p_hat <= 1.0
            assert y in (0.0, 0.5, 1.0)
    for row in csv_rows:
        assert set(row.keys()) == {"game_id", "quarter", "p_hat", "y"}


def test_game_rows_sorts_by_play_id():
    pbp = pd.DataFrame([
        {"game_id": "g1", "play_id": 3.0, "qtr": 1.0},
        {"game_id": "g1", "play_id": 1.0, "qtr": 1.0},
        {"game_id": "g2", "play_id": 2.0, "qtr": 1.0},
    ])
    result = game_rows(pbp, "g1")
    assert list(result["play_id"]) == [1.0, 3.0]
