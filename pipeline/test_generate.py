from dataclasses import replace

import torch
import pytest

import team_form as tf
from dataset import PlayDataset
from generate import GameSimulator, GameState
from model import GameTransformer


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


def _initial_state(sim):
    return GameState(
        quarter=1, play_in_quarter=0, down=1, ydstogo=10, yardline_100=75,
        posteam=sim.team_a, defteam=sim.team_b, posteam_score=0, defteam_score=0,
    )


def test_generate_runs_without_crashing_and_holds_invariants(simulator):
    generator = torch.Generator().manual_seed(1)
    log = simulator.generate(60, _initial_state(simulator), generator=generator)

    assert len(log) > 0
    for entry in log:
        s = entry["state"]
        assert 1 <= s.down <= 4
        assert 0 <= s.yardline_100 <= 100
        assert 1 <= s.ydstogo <= 100
        assert 1 <= s.quarter <= 5
        assert s.posteam_score >= 0 and s.defteam_score >= 0
        assert s.posteam != s.defteam
        assert {s.posteam, s.defteam} == {simulator.team_a, simulator.team_b}


def test_generate_stops_after_four_quarters(simulator):
    generator = torch.Generator().manual_seed(2)
    # far more plays than a real game has -- generate() must stop itself via
    # the quarter > 4 check rather than running forever.
    log = simulator.generate(2000, _initial_state(simulator), generator=generator)
    assert all(entry["state"].quarter <= 4 for entry in log)
    assert len(log) < 2000


def test_fourth_down_punts_when_out_of_range_and_not_short(simulator):
    # _initial_state's yardline_100=75/ydstogo=10 is out of FG range and not
    # short-yardage -- one specific case of the 4th-down decision, not a
    # universal "always punts" rule (see test_fourth_down_decision_* below
    # for the other branches of that decision).
    generator = torch.Generator().manual_seed(3)
    state = replace(_initial_state(simulator), down=4)
    log = simulator.generate(1, state, generator=generator)
    assert log[0]["event"] == "punt"
    assert log[0]["state"].down == 1
    assert log[0]["state"].posteam == simulator.team_b  # possession flipped


def test_fourth_down_decision_kicks_fg_in_range():
    from generate import GameState, _fourth_down_decision
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=8, yardline_100=25,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    assert _fourth_down_decision(state) == "fg"


def test_fourth_down_decision_punts_out_of_range_and_long():
    from generate import GameState, _fourth_down_decision
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=8, yardline_100=65,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    assert _fourth_down_decision(state) == "punt"


def test_fourth_down_decision_goes_for_it_short_yardage_out_of_range():
    from generate import GameState, _fourth_down_decision
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=1, yardline_100=65,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    assert _fourth_down_decision(state) == "go"


def test_fourth_down_decision_goes_for_it_when_desperate_regardless_of_distance():
    from generate import GameState, _fourth_down_decision
    state = GameState(quarter=4, play_in_quarter=10, down=4, ydstogo=8, yardline_100=65,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=10)
    assert _fourth_down_decision(state) == "go"


def test_fg_make_probability_short_kick_is_high():
    from generate import _fg_make_probability
    p = _fg_make_probability(kick_distance=25, kicker_fg_pct=None)
    assert p == pytest.approx(0.98)  # 20-29yd bucket, league-average kicker


def test_fg_make_probability_long_kick_is_low():
    from generate import _fg_make_probability
    p = _fg_make_probability(kick_distance=68, kicker_fg_pct=None)
    assert p == pytest.approx(0.37)  # 60+yd bucket, league-average kicker


def test_fg_make_probability_good_kicker_beats_average():
    from generate import _fg_make_probability, LEAGUE_AVG_FG_PCT
    p_avg = _fg_make_probability(kick_distance=45, kicker_fg_pct=None)
    p_good = _fg_make_probability(kick_distance=45, kicker_fg_pct=0.95)
    assert p_avg == pytest.approx(0.78)  # 40-49yd bucket baseline, no adjustment
    assert p_good == pytest.approx(0.78 + (0.95 - LEAGUE_AVG_FG_PCT))
    assert p_good > p_avg


def test_fg_make_probability_bucket_boundaries_are_correct():
    from generate import _fg_make_probability, FG_DISTANCE_BASELINE
    # edges: [19, 29, 39, 49, 59] -> buckets [<20, 20-29, 30-39, 40-49, 50-59, 60+]
    # each edge value itself lands in the bucket BELOW it (inclusive)
    cases = [
        (19, FG_DISTANCE_BASELINE[0]), (20, FG_DISTANCE_BASELINE[1]),
        (29, FG_DISTANCE_BASELINE[1]), (30, FG_DISTANCE_BASELINE[2]),
        (39, FG_DISTANCE_BASELINE[2]), (40, FG_DISTANCE_BASELINE[3]),
        (49, FG_DISTANCE_BASELINE[3]), (50, FG_DISTANCE_BASELINE[4]),
        (59, FG_DISTANCE_BASELINE[4]), (60, FG_DISTANCE_BASELINE[5]),
    ]
    for kick_distance, expected_baseline in cases:
        result = _fg_make_probability(kick_distance, kicker_fg_pct=None)
        assert result == pytest.approx(expected_baseline), f"kick_distance={kick_distance}"


def test_attempt_field_goal_certain_make_adds_three():
    import random
    from dataclasses import replace
    from generate import GameState, _attempt_field_goal
    random.seed(0)
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=3, yardline_100=1,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state, event = _attempt_field_goal(state, kicker_fg_pct=0.99)
    assert event == "field_goal_made"
    assert new_state.defteam_score == 3  # scoring team is now on defense (post-kickoff)


def test_attempt_field_goal_certain_miss_flips_possession_no_points():
    import random
    from generate import GameState, _attempt_field_goal
    random.seed(0)
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=3, yardline_100=65,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state, event = _attempt_field_goal(state, kicker_fg_pct=None)
    assert event == "field_goal_missed"
    assert new_state.posteam_score == 0 and new_state.defteam_score == 0
    assert new_state.posteam == "SF"  # possession flipped


def test_team_form_updates_across_rollout(simulator):
    # simulator is module-scoped and shared with earlier tests in this file, which
    # already ran generate() and mutated form_state -- reset so this test's own
    # "starts empty, ends populated" check is independent of test execution order.
    simulator.form_state = {}
    generator = torch.Generator().manual_seed(4)
    assert simulator.form_state == {}
    simulator.generate(10, _initial_state(simulator), generator=generator)
    assert simulator.form_state != {}
    assert simulator.team_a in simulator.form_state or simulator.team_b in simulator.form_state

    # A non-empty dict alone doesn't prove the state actually evolved from real
    # play outcomes -- confirm at least one team's OFFENSE side picked up real
    # history (both the yards-EMA and touchdown/turnover-EMA history flags),
    # which only happens once update_team_form has actually been fed a play.
    features_a = tf.team_form_features(simulator.form_state, simulator.team_a, simulator.team_b)
    features_b = tf.team_form_features(simulator.form_state, simulator.team_b, simulator.team_a)
    assert (features_a[3] == 1.0 and features_a[4] == 1.0) or (features_b[3] == 1.0 and features_b[4] == 1.0)


def test_touchdown_adds_seven_and_flips_possession_via_kickoff(simulator):
    # run several short rollouts and require at least one touchdown to show up
    # somewhere across them, then check its scoring/possession bookkeeping
    # against the immediately preceding state.
    for seed in range(10):
        initial = _initial_state(simulator)
        log = simulator.generate(30, initial, generator=torch.Generator().manual_seed(seed))
        prev_state = initial
        for entry in log:
            if entry["event"] == "touchdown":
                s = entry["state"]
                # the team that just scored is now the DEFENSE (post-kickoff,
                # possession flipped to the team that got scored on)
                assert s.defteam_score == prev_state.posteam_score + 7
                assert s.posteam_score == prev_state.defteam_score
                assert s.down == 1 and s.ydstogo == 10
                return
            prev_state = entry["state"]
    pytest.fail("expected at least one touchdown across sampled rollouts")


def test_team_form_credits_preplay_team_on_turnover(simulator, monkeypatch):
    # The single most subtle correctness point in the team-form wiring: a play's
    # outcome must be attributed to whichever team was on offense BEFORE the play
    # (state.posteam pre-play), not the team that ends up on offense afterward --
    # which, for a turnover, is the OPPOSITE team (possession flips).
    #
    # generate()'s log only records the state AFTER each play, and doesn't expose
    # intermediate form_state snapshots -- so instead of trying to replay/reconstruct
    # snapshots from outside, we spy directly on team_form.update_team_form (still
    # delegating to the real implementation) and record exactly what it was called
    # with each time. That's the actual call generate.py makes, so comparing those
    # recorded arguments against the log's pre-play state is a direct test of the
    # causal-ordering invariant, not an inference from side effects.
    real_update = tf.update_team_form
    calls = []

    def spy_update(form_state, posteam, defteam, *args, **kwargs):
        new_state = real_update(form_state, posteam, defteam, *args, **kwargs)
        calls.append({"posteam": posteam, "defteam": defteam, "before": form_state, "after": new_state})
        return new_state

    monkeypatch.setattr(tf, "update_team_form", spy_update)

    for seed in range(10):
        calls.clear()
        simulator.form_state = {}
        initial = _initial_state(simulator)
        log = simulator.generate(30, initial, generator=torch.Generator().manual_seed(seed))

        prev_state = initial
        call_idx = 0
        for entry in log:
            if entry["event"] == "punt":
                # punts short-circuit before the model/team_form step -- no call made
                prev_state = entry["state"]
                continue

            call = calls[call_idx]
            call_idx += 1

            # update_team_form must always be called with the state as it was
            # BEFORE this play (prev_state), never the state logged AFTER it.
            assert call["posteam"] == prev_state.posteam
            assert call["defteam"] == prev_state.defteam

            if entry["event"] == "turnover":
                post_state = entry["state"]
                # Possession has flipped in the log's post-play state. The credited
                # posteam must be the PRE-play team, which is now the post-play
                # DEFENSE -- not the post-play offense.
                assert call["posteam"] == post_state.defteam
                assert call["posteam"] != post_state.posteam

                # The pre-play team's own offense side picked up real history from
                # this exact call.
                credited_after = tf.team_form_features(call["after"], call["posteam"], call["defteam"])
                assert credited_after[3] == 1.0  # has_yards_history
                assert credited_after[4] == 1.0  # has_td_turnover_history

                # And the OTHER team (post-play posteam, i.e. the team that only
                # ends up on offense as a result of the turnover) must NOT have had
                # its offense side touched by this call -- proving the update went
                # to the team that actually ran the play, not the team the flip
                # left on offense afterward.
                other_before = tf.team_form_features(call["before"], call["defteam"], call["posteam"])[:5]
                other_after = tf.team_form_features(call["after"], call["defteam"], call["posteam"])[:5]
                assert list(other_before) == list(other_after)
                return

            prev_state = entry["state"]

    pytest.fail("expected at least one turnover across sampled rollouts")
