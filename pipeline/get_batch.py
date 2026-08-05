"""
get_batch, adapted from Karpathy's version: sample a random game, then a
random window of consecutive plays *within that game* -- a window must
never span two games (flagged as a hard requirement in PROJECT_BRIEF.md).

Unlike the char-level LM this is adapted from, there's no shift-by-one
between x and y: PlayDataset.__getitem__(i) already returns both the
situational/personnel *inputs* for play i and that same play's outcome
*targets* (masked by applicability). The causal mask inside the model is
what prevents position i from seeing play i+1..T when predicting play i's
outcome -- get_batch just needs to hand it a same-indexed window.
"""

import torch


def build_game_index(examples):
    """
    Maps game_id -> (start, end) exclusive index range into a flat examples
    list. Relies on build_dataset.build_examples() grouping plays by game_id
    contiguously (it iterates pbp.groupby("game_id"), which is sorted and
    contiguous by construction) -- this is NOT re-sorted here, so if that
    upstream ordering assumption ever breaks, ranges silently wouldn't line
    up. This function has no way to detect that; PlayDataset's callers own it.
    """
    ranges = {}
    start = 0
    prev_game = examples[0]["game_id"]
    for i, ex in enumerate(examples):
        if ex["game_id"] != prev_game:
            ranges[prev_game] = (start, i)
            start = i
            prev_game = ex["game_id"]
    ranges[prev_game] = (start, len(examples))
    return ranges


def _stack_leaf(windows, get_leaf):
    per_window = [torch.stack([get_leaf(item) for item in window]) for window in windows]  # each (T, ...)
    return torch.stack(per_window)  # (B, T, ...)


def collate_windows(windows):
    """windows: list (len B) of list (len T) of PlayDataset.__getitem__ dicts."""
    non_target_keys = [k for k in windows[0][0].keys() if k != "targets"]
    batch = {k: _stack_leaf(windows, lambda item, k=k: item[k]) for k in non_target_keys}
    target_keys = windows[0][0]["targets"].keys()
    batch["targets"] = {tk: _stack_leaf(windows, lambda item, tk=tk: item["targets"][tk]) for tk in target_keys}
    return batch


class GameBatcher:
    """Samples fixed-length, single-game windows from a PlayDataset for training."""

    def __init__(self, dataset, block_size):
        self.dataset = dataset
        self.block_size = block_size
        self.game_index = build_game_index(dataset.examples)
        self.game_ids = list(self.game_index.keys())
        self.playable_game_ids = [
            gid for gid, (start, end) in self.game_index.items() if end - start >= block_size
        ]
        if not self.playable_game_ids:
            raise ValueError(f"no game has >= block_size={block_size} plays; lower block_size")

    def sample_window_indices(self, generator=None):
        gid = self.playable_game_ids[
            torch.randint(len(self.playable_game_ids), (1,), generator=generator).item()
        ]
        start, end = self.game_index[gid]
        game_len = end - start
        offset = torch.randint(game_len - self.block_size + 1, (1,), generator=generator).item()
        window_start = start + offset
        return list(range(window_start, window_start + self.block_size))

    def get_batch(self, batch_size, generator=None):
        windows = [
            [self.dataset[i] for i in self.sample_window_indices(generator)]
            for _ in range(batch_size)
        ]
        return collate_windows(windows)
