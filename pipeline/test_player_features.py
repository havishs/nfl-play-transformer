import numpy as np
import pytest

from player_features import DEF_FEATS, OFF_FEATS, PlayerFeatureLookup, _order_key


@pytest.fixture(scope="module")
def lookup():
    return PlayerFeatureLookup([2022, 2023], data_dir="../data")


def test_history_sequence_shape(lookup):
    pid = next(iter(lookup._off_raw.keys()))
    seq, mask = lookup.lookup_history_sequence(pid, 2023, 10, max_history=16)
    assert seq.shape == (16, len(OFF_FEATS) + len(DEF_FEATS))
    assert mask.shape == (16,)
    assert not np.any(np.isnan(seq))


def test_history_sequence_no_future_leakage(lookup):
    pid = next(iter(lookup._off_raw.keys()))
    raw_weeks = sorted(lookup._off_raw[pid].index.tolist())
    cutoff_key = raw_weeks[len(raw_weeks) // 2]
    season, week = divmod(cutoff_key, 100)

    _, mask = lookup.lookup_history_sequence(pid, season, week, max_history=100)
    n_expected = sum(1 for k in raw_weeks if k < cutoff_key)
    assert mask.sum() == min(n_expected, 100)


def test_history_sequence_truncates_to_most_recent(lookup):
    pid = next(iter(lookup._off_raw.keys()))
    raw_weeks = sorted(lookup._off_raw[pid].index.tolist())
    assert len(raw_weeks) > 3, "test needs a player with more history than max_history"
    cutoff_key = raw_weeks[-1] + 1
    season, week = divmod(cutoff_key, 100)

    seq, mask = lookup.lookup_history_sequence(pid, season, week, max_history=3)
    assert mask.sum() == 3
    # the 3 real rows should be the 3 MOST RECENT weeks, not the earliest
    expected_weeks = set(raw_weeks[-3:])
    off_feat_cols = seq[mask][:, : len(OFF_FEATS)]
    raw_table = lookup._off_raw[pid]
    actual_rows = raw_table.loc[sorted(expected_weeks)][OFF_FEATS].fillna(0.0).to_numpy(dtype=np.float32)
    assert np.allclose(np.sort(off_feat_cols, axis=0), np.sort(actual_rows, axis=0))


def test_history_sequence_no_history_is_all_padding(lookup):
    pid = next(iter(lookup._off_raw.keys()))
    raw_weeks = sorted(lookup._off_raw[pid].index.tolist())
    earliest_key = raw_weeks[0]
    season, week = divmod(earliest_key, 100)

    seq, mask = lookup.lookup_history_sequence(pid, season, week, max_history=16)
    assert mask.sum() == 0
    assert np.all(seq == 0.0)


def test_history_sequence_unknown_player_is_all_padding(lookup):
    seq, mask = lookup.lookup_history_sequence("nonexistent_player_id", 2023, 10, max_history=16)
    assert mask.sum() == 0
    assert np.all(seq == 0.0)


def test_lookup_includes_player_id(lookup):
    pid = next(iter(lookup._off_raw.keys()))
    result = lookup.lookup(pid, 2023, 10)
    assert result["player_id"] == pid
