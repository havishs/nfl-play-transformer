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


def test_fg_pct_averages_all_same_week_attempts_not_just_the_last(lookup):
    # explicitly find a kicker with >1 FG attempt in the same real week --
    # this is the exact scenario that broke the originally-suggested
    # per-row expanding-mean approach (which only worked when each player
    # had at most one row per week, true for player_features.py's source
    # data but NOT true here). Don't rely on incidental dict-iteration
    # order to exercise this -- search for it directly.
    counts = lookup.fg.groupby(["kicker_player_id", "order_key"]).size()
    multi_attempt = counts[counts > 1]
    assert len(multi_attempt) > 0, "test needs at least one kicker with >1 same-week FG attempt in the real data"
    kicker_id, target_key = multi_attempt.index[0]
    season, week = divmod(int(target_key), 100)

    # find the next week (strictly later order_key) this kicker has ANY
    # attempt in, and confirm fg_pct at that point equals the mean of
    # ALL the target week's attempts, not just one of them.
    kicker_weeks = sorted(lookup.fg[lookup.fg["kicker_player_id"] == kicker_id]["order_key"].unique())
    later_weeks = [k for k in kicker_weeks if k > target_key]
    assert later_weeks, "test needs this kicker to have a later real attempt to query fg_pct at"
    query_key = later_weeks[0]
    query_season, query_week = divmod(int(query_key), 100)

    same_week_rows = lookup.fg[(lookup.fg["kicker_player_id"] == kicker_id) & (lookup.fg["order_key"] == target_key)]
    prior_to_query = lookup.fg[(lookup.fg["kicker_player_id"] == kicker_id) & (lookup.fg["order_key"] < query_key)]
    expected = float(prior_to_query["made"].mean())

    result = lookup.fg_pct(kicker_id, query_season, query_week)
    assert result == pytest.approx(expected)
    assert len(same_week_rows) > 1  # sanity: confirms this row group really is a multi-attempt week


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


def test_primary_punter_returns_a_real_id_used_that_season(lookup):
    team = lookup.punt["posteam"].iloc[0]
    season = int(lookup.punt["season"].iloc[0])
    punter_id = lookup.primary_punter(team, season)
    assert punter_id is not None
    assert punter_id in set(lookup.punt[(lookup.punt["posteam"] == team) & (lookup.punt["season"] == season)]["punter_player_id"])


def test_primary_punter_unknown_team_is_none(lookup):
    assert lookup.primary_punter("ZZZ", 2023) is None


def test_order_key_matches_player_features_convention():
    assert _order_key(2023, 5) == 202305
    assert _order_key(2022, 17) < _order_key(2023, 1)
