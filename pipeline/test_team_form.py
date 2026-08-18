import numpy as np
import pytest

from team_form import EMA_ALPHA, initial_team_form, team_form_features, update_team_form


def test_bootstrap_sets_ema_directly_not_blended_from_zero():
    state = initial_team_form()
    state = update_team_form(state, "KC", "SF", yards_gained=10.0, yards_applicable=True,
                              touchdown=False, turnover=False, td_turnover_applicable=True)
    feats = team_form_features(state, "KC", "SF")
    assert feats[0] == 10.0  # KC offense yards_ema == the single observed value, not EMA_ALPHA*10


def test_second_update_blends_with_ema_alpha():
    state = initial_team_form()
    state = update_team_form(state, "KC", "SF", 10.0, True, False, False, True)
    state = update_team_form(state, "KC", "SF", 0.0, True, False, False, True)
    feats = team_form_features(state, "KC", "SF")
    expected = EMA_ALPHA * 0.0 + (1 - EMA_ALPHA) * 10.0
    assert feats[0] == pytest.approx(expected)


def test_yards_only_play_does_not_touch_touchdown_turnover_flags():
    state = initial_team_form()
    # a qb_kneel: yards-applicable, NOT touchdown/turnover-applicable
    state = update_team_form(state, "KC", "SF", yards_gained=-1.0, yards_applicable=True,
                              touchdown=None, turnover=None, td_turnover_applicable=False)
    feats = team_form_features(state, "KC", "SF")
    assert feats[3] == 1.0  # has_yards_history
    assert feats[4] == 0.0  # has_td_turnover_history still False
    assert feats[1] == 0.0  # touchdown_ema untouched (still default)
    assert feats[2] == 0.0  # turnover_ema untouched (still default)


def test_offense_and_defense_sides_are_independent():
    state = initial_team_form()
    state = update_team_form(state, "KC", "SF", 20.0, True, True, False, True)
    feats = team_form_features(state, "KC", "SF")
    # KC's OFFENSE side updated (posteam)
    assert feats[0] == 20.0
    assert feats[1] == 1.0
    # SF's DEFENSE side updated (defteam) -- same play, same values, different side
    assert feats[5] == 20.0
    assert feats[6] == 1.0
    # KC has no defense history yet, SF has no offense history yet
    reverse_feats = team_form_features(state, "SF", "KC")  # SF now posteam, KC now defteam
    assert reverse_feats[3] == 0.0  # SF's OFFENSE side untouched
    assert reverse_feats[8] == 0.0  # KC's DEFENSE side untouched


def test_unrelated_team_has_no_history():
    state = initial_team_form()
    state = update_team_form(state, "KC", "SF", 20.0, True, False, False, True)
    feats = team_form_features(state, "DAL", "NYG")
    assert np.all(feats == 0.0)
