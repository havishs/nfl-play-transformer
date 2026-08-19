import numpy as np
import pytest

from special_teams_features import SpecialTeamsFeatureLookup, _order_key


@pytest.fixture(scope="module")
def lookup():
    return SpecialTeamsFeatureLookup([2022, 2023], data_dir="../data")


def test_fg_pct_no_future_leakage(lookup):
    kicker_id = next(iter(lookup._fg_cache.keys()))
    weeks = sorted(lookup._fg_cache[kicker_id].index.tolist())
    assert len(weeks) > 1, "test needs a kicker with more than one tracked week"
    cutoff_key = weeks[len(weeks) // 2]
    season, week = divmod(cutoff_key, 100)

    result = lookup.fg_pct(kicker_id, season, week)
    prior_rows = lookup._fg_cache[kicker_id][lookup._fg_cache[kicker_id].index < cutoff_key]
    expected = float(prior_rows["made"].mean()) if len(prior_rows) else None
    if expected is None or np.isnan(expected):
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_fg_pct_no_prior_attempts_is_none(lookup):
    kicker_id = next(iter(lookup._fg_cache.keys()))
    weeks = sorted(lookup._fg_cache[kicker_id].index.tolist())
    earliest_key = weeks[0]
    season, week = divmod(earliest_key, 100)

    assert lookup.fg_pct(kicker_id, season, week) is None


def test_fg_pct_unknown_kicker_is_none(lookup):
    assert lookup.fg_pct("nonexistent_kicker_id", 2023, 10) is None


def test_punt_avg_distance_no_future_leakage(lookup):
    punter_id = next(iter(lookup._punt_cache.keys()))
    weeks = sorted(lookup._punt_cache[punter_id].index.tolist())
    assert len(weeks) > 1, "test needs a punter with more than one tracked week"
    cutoff_key = weeks[len(weeks) // 2]
    season, week = divmod(cutoff_key, 100)

    result = lookup.punt_avg_distance(punter_id, season, week)
    prior_rows = lookup._punt_cache[punter_id][lookup._punt_cache[punter_id].index < cutoff_key]
    expected = float(prior_rows["kick_distance"].mean()) if len(prior_rows) else None
    if expected is None or np.isnan(expected):
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_punt_avg_distance_unknown_punter_is_none(lookup):
    assert lookup.punt_avg_distance("nonexistent_punter_id", 2023, 10) is None


def test_primary_kicker_and_punter_return_a_real_id_used_that_season(lookup):
    team = lookup.fg["posteam"].iloc[0]
    season = int(lookup.fg["season"].iloc[0])
    kicker_id = lookup.primary_kicker(team, season)
    assert kicker_id is not None
    assert kicker_id in set(lookup.fg[(lookup.fg["posteam"] == team) & (lookup.fg["season"] == season)]["kicker_player_id"])


def test_primary_kicker_unknown_team_is_none(lookup):
    assert lookup.primary_kicker("ZZZ", 2023) is None


def test_order_key_matches_player_features_convention():
    assert _order_key(2023, 5) == 202305
    assert _order_key(2022, 17) < _order_key(2023, 1)
