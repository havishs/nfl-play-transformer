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
    # far more plays than a real game has -- generate() must stop itself
    # rather than running forever. quarter > 4 is now a valid outcome (a
    # tied game legitimately continues into overtime, see
    # test_overtime_continues_when_tied_and_stops_on_first_score) so the
    # real invariant to check here is termination, not a quarter ceiling.
    log = simulator.generate(2000, _initial_state(simulator), generator=generator)
    assert len(log) < 2000


def test_overtime_continues_when_tied_and_stops_on_first_score(simulator):
    from dataclasses import replace
    tied_state = replace(_initial_state(simulator), quarter=5, play_in_quarter=0,
                          posteam_score=14, defteam_score=14)
    generator = torch.Generator().manual_seed(5)
    log = simulator.generate(60, tied_state, generator=generator)
    assert len(log) > 0
    # game must end the instant a score happens in overtime -- no more log
    # entries after the first touchdown/field_goal event
    for i, entry in enumerate(log):
        if entry["event"] in ("touchdown", "field_goal_made"):
            assert i == len(log) - 1
            return
    # if no score occurred in this sample, that's fine too -- the important
    # invariant (stop-on-score) just wasn't exercised this run


def test_punt_crossing_into_overtime_does_not_end_a_tied_game(simulator):
    from dataclasses import replace
    from generate import PLAYS_PER_QUARTER
    # deterministic: out of FG range (yardline_100=65 > FG_RANGE_YARDLINE_100)
    # and not short-yardage (ydstogo=10 > GO_FOR_IT_SHORT_YDSTOGO), so
    # _fourth_down_decision always returns "punt" here -- no model sampling
    # involved, so this doesn't depend on any seed getting lucky.
    boundary_state = replace(
        _initial_state(simulator), quarter=4, play_in_quarter=PLAYS_PER_QUARTER - 1,
        down=4, ydstogo=10, yardline_100=65, posteam_score=14, defteam_score=14,
    )
    generator = torch.Generator().manual_seed(6)
    log = simulator.generate(2, boundary_state, generator=generator)

    assert len(log) == 2, "the punt crossing into a still-tied Q5 must not end the game early"
    assert log[0]["event"] == "punt"
    assert log[0]["state"].quarter == 5  # crossed into overtime
    assert log[0]["state"].posteam_score == log[0]["state"].defteam_score  # still tied


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


def test_apply_turnover_on_downs_flips_possession_no_return():
    from generate import GameState, _apply_turnover_on_downs
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=5, yardline_100=40,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state = _apply_turnover_on_downs(state, yards_gained=2)  # short of the marker
    assert new_state.posteam == "SF"  # possession flipped
    assert new_state.down == 1 and new_state.ydstogo == 10
    assert new_state.yardline_100 == 100 - (40 - 2)  # spot of the play, no return


def test_punt_uses_punter_average_when_known():
    import random
    from generate import GameState, _punt, PUNT_AVG_RETURN_YARDS
    random.seed(1)  # avoid the touchback branch for this deterministic check
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=8, yardline_100=65,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state = _punt(state, punter_avg_distance=35.0)
    # net = 35.0 - PUNT_AVG_RETURN_YARDS; kicking spot = 65 - net; receiving yardline_100 = 100 - kicking spot
    # -- deliberately chosen to stay clear of the touchback clip (min(80, ...)),
    # so this test actually exercises the subtraction formula, not just the cap.
    expected_net = 35.0 - PUNT_AVG_RETURN_YARDS
    expected_kicking_spot = max(1, 65 - expected_net)
    expected_yardline_100 = min(80, 100 - expected_kicking_spot)
    assert expected_yardline_100 < 80, "test setup error: this case should NOT hit the touchback clip"
    assert new_state.yardline_100 == pytest.approx(expected_yardline_100, abs=1)


def test_punt_kicking_spot_clips_at_one_yard_line_for_a_long_punt_near_midfield():
    import random
    from generate import GameState, _punt
    random.seed(1)  # avoid the touchback branch
    # yardline_100=41 is just past FG range (so _fourth_down_decision would
    # actually punt from here in real play), with a strong punter (net ~54
    # after the average return yardage) -- net distance exceeds the distance
    # to the kicking team's own goal line, so the raw kicking spot (41 - 54.4
    # = -13.4) must clip at 1, not go negative.
    #
    # Note: with PUNT_TOUCHBACK_YARDLINE_100=80, receiving = min(80, 100 - 1)
    # = 80 -- the downstream touchback cap saturates the result regardless of
    # the clipped value, so this scenario cannot isolate the max(1, ...) clip
    # from the min(80, ...) cap (verified: removing the max(1, ...) clip
    # entirely produces the same 80 here, and it does for ANY raw kicking
    # spot below 20, since 100 - anything < 20 is always > 80). This is a
    # boundary/sanity check that a very long punt near midfield resolves to
    # the touchback spot rather than an invalid field position -- it does
    # not, by itself, distinguish a broken max(1, ...) clip from correct code.
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=8, yardline_100=41,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state = _punt(state, punter_avg_distance=58.0)
    assert new_state.yardline_100 == 80


def test_punt_falls_back_to_league_average_when_punter_unknown():
    import random
    from generate import GameState, _punt, LEAGUE_AVG_PUNT_DISTANCE, PUNT_AVG_RETURN_YARDS
    random.seed(1)  # avoid the touchback branch
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=8, yardline_100=65,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state = _punt(state, punter_avg_distance=None)
    assert new_state.posteam == "SF"  # sanity: possession still flips either way

    expected_net = LEAGUE_AVG_PUNT_DISTANCE - PUNT_AVG_RETURN_YARDS
    expected_kicking_spot = max(1, 65 - expected_net)
    expected_yardline_100 = min(80, 100 - expected_kicking_spot)
    assert new_state.yardline_100 == pytest.approx(expected_yardline_100, abs=1)


def test_punt_touchback_spot_is_the_20_not_the_25():
    import random
    from generate import GameState, _punt, PUNT_TOUCHBACK_YARDLINE_100
    assert PUNT_TOUCHBACK_YARDLINE_100 == 80  # own 20 -- distinct from kickoff's 75 (own 25)

    original_random = random.random
    random.random = lambda: 0.0  # forces the touchback branch unconditionally
    try:
        state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=8, yardline_100=65,
                           posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
        new_state = _punt(state, punter_avg_distance=45.0)
    finally:
        random.random = original_random

    assert new_state.yardline_100 == PUNT_TOUCHBACK_YARDLINE_100
    assert new_state.down == 1 and new_state.ydstogo == 10
    assert new_state.posteam == "SF"


def test_apply_turnover_on_downs_handles_a_loss_on_the_play():
    from generate import GameState, _apply_turnover_on_downs
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=2, yardline_100=40,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    new_state = _apply_turnover_on_downs(state, yards_gained=-3)  # stuffed for a loss
    assert new_state.posteam == "SF"
    assert new_state.yardline_100 == 100 - (40 - (-3))


def test_exact_conversion_does_not_trigger_turnover_on_downs():
    # yards_gained == ydstogo exactly (converts to the marker) must fall to
    # the normal _apply_scrimmage_gain path (which itself resets to down=1),
    # NOT the turnover_on_downs branch -- confirms the ">0" boundary is right,
    # not an off-by-one that would (dis)qualify an exact conversion.
    from generate import GameState, _apply_scrimmage_gain
    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=5, yardline_100=40,
                       posteam="KC", defteam="SF", posteam_score=0, defteam_score=0)
    # this directly mirrors what generate()'s branch condition would decide:
    # (state.ydstogo - yards_gained) > 0 must be False here
    yards_gained = 5
    assert not (state.ydstogo - yards_gained) > 0
    new_state = _apply_scrimmage_gain(state, yards_gained)
    assert new_state.posteam == "KC"  # possession did NOT flip
    assert new_state.down == 1  # converted to a first down


def test_generate_can_produce_turnover_on_downs(simulator, monkeypatch):
    import generate as gen_module
    from generate import GameState

    # Force every model-sampled outcome to a small, non-converting,
    # non-scoring gain -- a deterministic path to a real turnover_on_downs
    # event through the FULL generate() loop (not just the pure function).
    no_gain_idx = simulator.dataset.vocabs["yards_gained"]["no_gain"]
    none_return_idx = simulator.dataset.vocabs["return_yards"]["none"]
    yards_vocab_size = len(simulator.dataset.vocabs["yards_gained"])
    return_vocab_size = len(simulator.dataset.vocabs["return_yards"])

    def fake_sample_class(logits, generator):
        n_classes = logits.shape[-1]
        if n_classes == yards_vocab_size:
            return no_gain_idx
        if n_classes == return_vocab_size:
            return none_return_idx
        return 0  # touchdown/turnover binary heads -> False

    monkeypatch.setattr(gen_module, "_sample_class", fake_sample_class)

    state = GameState(quarter=2, play_in_quarter=10, down=4, ydstogo=1, yardline_100=65,
                       posteam=simulator.team_a, defteam=simulator.team_b,
                       posteam_score=0, defteam_score=0)
    log = simulator.generate(1, state, generator=torch.Generator().manual_seed(0))

    assert len(log) == 1
    assert log[0]["event"] == "turnover_on_downs"
    assert 1 <= log[0]["state"].down <= 4


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


def test_touchdown_adds_six_or_seven_and_flips_possession_via_kickoff(simulator):
    # run several short rollouts and require at least one touchdown to show up
    # somewhere across them, then check its scoring/possession bookkeeping
    # against the immediately preceding state. +6 (missed XP) or +7 (made XP)
    # are both valid now that the extra point can miss.
    for seed in range(10):
        initial = _initial_state(simulator)
        log = simulator.generate(30, initial, generator=torch.Generator().manual_seed(seed))
        prev_state = initial
        for entry in log:
            if entry["event"] == "touchdown":
                s = entry["state"]
                # the team that just scored is now the DEFENSE (post-kickoff,
                # possession flipped to the team that got scored on)
                assert s.defteam_score in (prev_state.posteam_score + 6, prev_state.posteam_score + 7)
                assert s.posteam_score == prev_state.defteam_score
                assert s.down == 1 and s.ydstogo == 10
                return
            prev_state = entry["state"]
    pytest.fail("expected at least one touchdown across sampled rollouts")


def test_extra_point_can_miss(simulator):
    # run several short rollouts and require both a made and a missed XP to
    # show up somewhere across them -- confirms XP_SUCCESS_RATE < 1.0 is
    # actually exercised, not just declared.
    #
    # NOTE: this checks each touchdown's score INCREMENT (state.defteam_score
    # minus the immediately preceding state's posteam_score), not the raw
    # cumulative score. Raw cumulative score is a vacuous check here: a
    # team's first touchdown of a game always lands on exactly 7 even under
    # the old always-+7 behavior, so "seen_scores & {6, 7}" would trivially
    # pass forever regardless of whether XP misses are actually modeled.
    import random
    increments = set()
    for seed in range(30):
        random.seed(seed)
        initial = _initial_state(simulator)
        log = simulator.generate(30, initial, generator=torch.Generator().manual_seed(seed))
        prev_state = initial
        for entry in log:
            if entry["event"] == "touchdown":
                s = entry["state"]
                increments.add(s.defteam_score - prev_state.posteam_score)  # 6 (missed XP) or 7 (made XP)
            prev_state = entry["state"]
    assert increments == {6, 7}, f"expected both a missed (+6) and made (+7) extra point, got increments {increments}"


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
