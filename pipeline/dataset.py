"""
torch.utils.data.Dataset wrapping build_dataset.build_examples() output into
tensors: situational fields as category indices, offense/defense personnel
as (11, feature_dim) position-index + continuous-feature tensors, and
targets as indices with applicability masks (per PROJECT_BRIEF.md, targets
marked not-applicable must be masked out of loss, never trained as a fake
zero).

Scope: this is step 1 of the plan only -- indices + masks. No DataLoader /
get_batch, no model code yet.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

import vocab as V
from build_dataset import build_examples
from player_features import PlayerFeatureLookup

N_PERSONNEL = 11  # per side. See _pad_personnel: real data is occasionally off this count.

SITUATIONAL_FIELDS = ["play_type", "down", "distance", "field_zone", "score_diff", "quarter", "time_bucket"]
TEAM_FIELDS = ["posteam", "defteam"]

BIO_KEYS = ["years_exp", "draft_number"]
OFF_KEYS = ["passing_epa", "rushing_epa", "receiving_epa", "target_share"]
DEF_KEYS = ["def_completion_pct", "def_passer_rating_allowed", "def_pressures", "def_sacks", "def_missed_tackle_pct"]
# bio + off + def + has_off_stats + has_def_stats + is_padding
PERSONNEL_FEATURE_DIM = len(BIO_KEYS) + len(OFF_KEYS) + len(DEF_KEYS) + 3

MAX_HISTORY = 16  # ~1 season of games, per player, for the learned history encoder
HISTORY_FEATURE_DIM = len(OFF_KEYS) + len(DEF_KEYS)


def _pad_personnel(players, n=N_PERSONNEL):
    """
    Source personnel lists are 11 per side on the overwhelming majority of
    plays but occasionally off (data glitches -- 9 to 13 observed in real
    2023 pbp data). Truncate extras, pad shortfalls with an explicit None
    slot (rendered as zero features + is_padding=1) rather than crashing or
    silently misaligning the tensor. Counts are logged once in
    PlayDataset.__init__, not silently absorbed.
    """
    players = list(players[:n])
    players += [None] * (n - len(players))
    return players


def _personnel_slot_to_vectors(player, position_vocab):
    if player is None:
        feats = np.zeros(PERSONNEL_FEATURE_DIM, dtype=np.float32)
        feats[-1] = 1.0  # is_padding
        return position_vocab["UNK"], feats

    pos_idx = position_vocab.get(player["position"], position_vocab["UNK"])
    bio = np.nan_to_num([float(player["bio"][k]) for k in BIO_KEYS], nan=0.0)
    off = [player["off_feats"][k] for k in OFF_KEYS]
    defn = [player["def_feats"][k] for k in DEF_KEYS]
    flags = [float(player["has_off_stats"]), float(player["has_def_stats"]), 0.0]  # is_padding = 0
    feats = np.concatenate([bio, off, defn, flags]).astype(np.float32)
    return pos_idx, feats


class PlayDataset(Dataset):
    def __init__(self, history_seasons, training_seasons, data_dir="../data", max_examples=None, cache_path=None):
        """
        cache_path: if given and the file exists, load (examples, vocabs)
        from there instead of rebuilding -- build_examples()'s per-row
        iterrows() loop dominates wall time on multi-season data (minutes,
        not seconds), so repeated experiments on the same season config
        benefit from paying that cost once. Caller is responsible for using
        a path that reflects the season config (e.g. encoding the seasons
        in the filename) -- this class does not validate that a cached file
        actually matches the requested seasons.
        """
        vocab_seasons = sorted(set(history_seasons) | set(training_seasons))
        # Separate from build_examples()'s own internal PlayerFeatureLookup (which
        # only returns already-collapsed mean features) -- this one serves raw
        # per-week history sequences for the learned history encoder, looked up
        # lazily in __getitem__. Memoized below since many plays in the same game
        # request the identical (player_id, season, week) sequence.
        self.history_lookup = PlayerFeatureLookup(vocab_seasons, data_dir)
        self._history_cache = {}

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.examples, self.vocabs = pickle.load(f)
            self._log_personnel_count_anomalies()
            return

        self.examples = build_examples(history_seasons, training_seasons, data_dir, max_examples)
        self.vocabs = V.build_vocabs(vocab_seasons, data_dir)
        self._log_personnel_count_anomalies()

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump((self.examples, self.vocabs), f)

    def _log_personnel_count_anomalies(self):
        off_bad = sum(1 for ex in self.examples if len(ex["offense"]) != N_PERSONNEL)
        def_bad = sum(1 for ex in self.examples if len(ex["defense"]) != N_PERSONNEL)
        if off_bad or def_bad:
            print(f"PlayDataset: personnel count != {N_PERSONNEL} on "
                  f"{off_bad} offense / {def_bad} defense plays (padded short, truncated long)")

    def __len__(self):
        return len(self.examples)

    def _situational_tensor(self, situational):
        idxs = [self.vocabs[f][situational[f]] for f in SITUATIONAL_FIELDS]
        idxs += [self.vocabs["team"].get(situational[f], self.vocabs["team"]["UNK"]) for f in TEAM_FIELDS]
        return torch.tensor(idxs, dtype=torch.long)

    def _personnel_tensors(self, players):
        padded = _pad_personnel(players)
        pos_idxs, feats = zip(*(_personnel_slot_to_vectors(p, self.vocabs["position"]) for p in padded))
        return torch.tensor(pos_idxs, dtype=torch.long), torch.tensor(np.stack(feats), dtype=torch.float32)

    def _get_history(self, player_id, season, week):
        key = (player_id, season, week)
        if key not in self._history_cache:
            self._history_cache[key] = self.history_lookup.lookup_history_sequence(
                player_id, season, week, MAX_HISTORY
            )
        return self._history_cache[key]

    def _history_tensors(self, players, season, week):
        padded = _pad_personnel(players)
        seqs, masks = [], []
        for p in padded:
            if p is None:
                seqs.append(np.zeros((MAX_HISTORY, HISTORY_FEATURE_DIM), dtype=np.float32))
                masks.append(np.zeros(MAX_HISTORY, dtype=bool))
            else:
                seq, mask = self._get_history(p["player_id"], season, week)
                seqs.append(seq)
                masks.append(mask)
        return torch.tensor(np.stack(seqs), dtype=torch.float32), torch.tensor(np.stack(masks), dtype=torch.bool)

    def _target_tensors(self, targets):
        def idx_and_mask(value, vocab_name):
            if value is None:
                return 0, 0.0
            return self.vocabs[vocab_name][value], 1.0

        yards_idx, yards_mask = idx_and_mask(targets["yards_gained"], "yards_gained")
        return_idx, return_mask = idx_and_mask(targets["return_yards"], "return_yards")
        td_turnover_mask = 1.0 if targets["td_turnover_applicable"] else 0.0
        touchdown = int(targets["touchdown"]) if targets["touchdown"] is not None else 0
        turnover = int(targets["turnover"]) if targets["turnover"] is not None else 0

        return {
            "yards_gained": torch.tensor(yards_idx, dtype=torch.long),
            "yards_gained_mask": torch.tensor(yards_mask, dtype=torch.float32),
            "touchdown": torch.tensor(touchdown, dtype=torch.long),
            "turnover": torch.tensor(turnover, dtype=torch.long),
            "td_turnover_mask": torch.tensor(td_turnover_mask, dtype=torch.float32),
            "return_yards": torch.tensor(return_idx, dtype=torch.long),
            "return_yards_mask": torch.tensor(return_mask, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        ex = self.examples[idx]
        season = int(ex["game_id"].split("_")[0])
        week = ex["week"]
        off_pos, off_feats = self._personnel_tensors(ex["offense"])
        def_pos, def_feats = self._personnel_tensors(ex["defense"])
        off_hist, off_hist_mask = self._history_tensors(ex["offense"], season, week)
        def_hist, def_hist_mask = self._history_tensors(ex["defense"], season, week)
        return {
            "situational": self._situational_tensor(ex["situational"]),
            "offense_position": off_pos,
            "offense_features": off_feats,
            "offense_history": off_hist,
            "offense_history_mask": off_hist_mask,
            "defense_position": def_pos,
            "defense_features": def_feats,
            "defense_history": def_hist,
            "defense_history_mask": def_hist_mask,
            "targets": self._target_tensors(ex["targets"]),
        }
