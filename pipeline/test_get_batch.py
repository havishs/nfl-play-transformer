import pytest
import torch

from dataset import N_PERSONNEL, PERSONNEL_FEATURE_DIM, PlayDataset
from get_batch import GameBatcher, build_game_index


@pytest.fixture(scope="module")
def multi_game_dataset():
    # ~1500 examples spans roughly 8-9 games at ~174 plays/game -- enough to
    # exercise cross-game boundaries without loading the full season.
    return PlayDataset(history_seasons=[2022], training_seasons=[2023], data_dir="../data", max_examples=1500)


def test_build_game_index_partitions_examples_exactly(multi_game_dataset):
    examples = multi_game_dataset.examples
    index = build_game_index(examples)

    covered = 0
    for gid, (start, end) in index.items():
        assert end > start
        assert all(examples[i]["game_id"] == gid for i in range(start, end))
        covered += end - start
    assert covered == len(examples)

    # ranges don't overlap and union covers [0, len(examples))
    ranges = sorted(index.values())
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(examples)
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert prev_end == next_start


def test_sampled_windows_never_span_two_games(multi_game_dataset):
    batcher = GameBatcher(multi_game_dataset, block_size=16)
    examples = multi_game_dataset.examples
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        indices = batcher.sample_window_indices(generator)
        assert len(indices) == 16
        game_ids = {examples[i]["game_id"] for i in indices}
        assert len(game_ids) == 1
        assert indices == list(range(indices[0], indices[0] + 16))  # contiguous


def test_get_batch_shapes(multi_game_dataset):
    batch_size, block_size = 4, 16
    batcher = GameBatcher(multi_game_dataset, block_size=block_size)
    batch = batcher.get_batch(batch_size, generator=torch.Generator().manual_seed(1))

    assert batch["situational"].shape == (batch_size, block_size, 9)
    assert batch["offense_position"].shape == (batch_size, block_size, N_PERSONNEL)
    assert batch["offense_features"].shape == (batch_size, block_size, N_PERSONNEL, PERSONNEL_FEATURE_DIM)
    assert batch["defense_position"].shape == (batch_size, block_size, N_PERSONNEL)
    assert batch["defense_features"].shape == (batch_size, block_size, N_PERSONNEL, PERSONNEL_FEATURE_DIM)
    assert batch["targets"]["yards_gained"].shape == (batch_size, block_size)
    assert batch["targets"]["yards_gained_mask"].shape == (batch_size, block_size)


def test_game_batcher_rejects_block_size_larger_than_every_game(multi_game_dataset):
    with pytest.raises(ValueError):
        GameBatcher(multi_game_dataset, block_size=100_000)
