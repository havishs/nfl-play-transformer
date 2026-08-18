import numpy as np
import pytest
import torch

import situational as sit
import vocab as V
from dataset import (
    N_PERSONNEL,
    PERSONNEL_FEATURE_DIM,
    TEAM_FORM_FEATURE_DIM,
    PlayDataset,
    _pad_personnel,
    _personnel_slot_to_vectors,
)

ALL_BUCKET_LISTS = {
    "play_type": (sit.PLAY_TYPES, V.PLAY_TYPE_VOCAB),
    "down": (sit.DOWN_BUCKETS, V.DOWN_VOCAB),
    "distance": (sit.DISTANCE_BUCKETS, V.DISTANCE_VOCAB),
    "field_zone": (sit.FIELD_ZONE_BUCKETS, V.FIELD_ZONE_VOCAB),
    "score_diff": (sit.SCORE_DIFF_BUCKETS, V.SCORE_DIFF_VOCAB),
    "quarter": (sit.QUARTER_BUCKETS, V.QUARTER_VOCAB),
    "time_bucket": (sit.TIME_BUCKETS, V.TIME_VOCAB),
    "yards_gained": (sit.YARDS_GAINED_BUCKETS, V.YARDS_GAINED_VOCAB),
    "return_yards": (sit.RETURN_YARDS_BUCKETS, V.RETURN_YARDS_VOCAB),
}


@pytest.mark.parametrize("field", ALL_BUCKET_LISTS.keys())
def test_vocab_covers_every_bucket_label(field):
    labels, built_vocab = ALL_BUCKET_LISTS[field]
    assert set(built_vocab.keys()) == set(labels)
    assert sorted(built_vocab.values()) == list(range(len(labels)))


def test_team_and_position_vocab_include_unk():
    teams = V.build_team_vocab([2023], data_dir="../data")
    positions = V.build_position_vocab([2023], data_dir="../data")
    assert "UNK" in teams
    assert "UNK" in positions
    assert len(teams) == 33  # 32 real teams + UNK, checked against real 2022/2023 data
    assert len(positions) == 12  # 11 real positions + UNK


def test_pad_personnel_truncates_extra_players():
    players = list(range(13))  # more than N_PERSONNEL
    padded = _pad_personnel(players)
    assert len(padded) == N_PERSONNEL
    assert padded == list(range(N_PERSONNEL))


def test_pad_personnel_pads_missing_with_none():
    players = list(range(9))  # fewer than N_PERSONNEL
    padded = _pad_personnel(players)
    assert len(padded) == N_PERSONNEL
    assert padded[:9] == list(range(9))
    assert padded[9:] == [None, None]


def test_personnel_slot_padding_marks_is_padding_and_unk_position():
    position_vocab = {"UNK": 5}
    pos_idx, feats = _personnel_slot_to_vectors(None, position_vocab)
    assert pos_idx == position_vocab["UNK"]
    assert feats.shape == (PERSONNEL_FEATURE_DIM,)
    assert feats[-1] == 1.0  # is_padding
    assert np.all(feats[:-1] == 0.0)


def test_personnel_slot_real_player_not_marked_padding():
    position_vocab = {"QB": 0, "UNK": 1}
    player = {
        "position": "QB",
        "bio": {"years_exp": 4, "draft_number": np.nan},
        "off_feats": {"passing_epa": 0.1, "rushing_epa": 0.0, "receiving_epa": 0.0, "target_share": 0.0},
        "def_feats": {"def_completion_pct": 0.0, "def_passer_rating_allowed": 0.0,
                      "def_pressures": 0.0, "def_sacks": 0.0, "def_missed_tackle_pct": 0.0},
        "has_off_stats": True,
        "has_def_stats": False,
    }
    pos_idx, feats = _personnel_slot_to_vectors(player, position_vocab)
    assert pos_idx == position_vocab["QB"]
    assert feats[-1] == 0.0  # is_padding
    assert feats[1] == 0.0  # NaN draft_number zero-filled, not left as NaN
    assert not np.any(np.isnan(feats))


@pytest.fixture(scope="module")
def small_dataset():
    return PlayDataset(history_seasons=[2022], training_seasons=[2023], data_dir="../data", max_examples=100)


def test_dataset_item_shapes(small_dataset):
    ex = small_dataset[0]
    assert ex["situational"].shape == (9,)
    assert ex["situational"].dtype == torch.long
    assert ex["offense_position"].shape == (N_PERSONNEL,)
    assert ex["offense_features"].shape == (N_PERSONNEL, PERSONNEL_FEATURE_DIM)
    assert ex["defense_position"].shape == (N_PERSONNEL,)
    assert ex["defense_features"].shape == (N_PERSONNEL, PERSONNEL_FEATURE_DIM)


def test_team_form_shape_and_first_play_has_no_history(small_dataset):
    ex = small_dataset[0]
    assert ex["team_form"].shape == (TEAM_FORM_FEATURE_DIM,)
    assert ex["team_form"].dtype == torch.float32
    # first play of the dataset's first game -- neither team has any prior
    # plays in this game yet, so both cold-start flags must be 0.
    assert ex["team_form"][3].item() == 0.0  # posteam has_yards_history
    assert ex["team_form"][4].item() == 0.0  # posteam has_td_turnover_history
    assert ex["team_form"][8].item() == 0.0  # defteam has_yards_history
    assert ex["team_form"][9].item() == 0.0  # defteam has_td_turnover_history


def test_dataset_masks_are_zero_or_one(small_dataset):
    for i in range(len(small_dataset)):
        targets = small_dataset[i]["targets"]
        for mask_key in ("yards_gained_mask", "td_turnover_mask", "return_yards_mask"):
            assert targets[mask_key].item() in (0.0, 1.0)


def test_dataset_masked_out_target_uses_dummy_index_zero(small_dataset):
    # find at least one example where return_yards isn't applicable (e.g. a run play)
    # and confirm the dummy index is 0, not a real bucket that could be mistaken for signal.
    found = False
    for i in range(len(small_dataset)):
        targets = small_dataset[i]["targets"]
        if targets["return_yards_mask"].item() == 0.0:
            assert targets["return_yards"].item() == 0
            found = True
    assert found, "expected at least one example with return_yards not applicable"


def test_dataset_length_matches_examples(small_dataset):
    assert len(small_dataset) == len(small_dataset.examples)
