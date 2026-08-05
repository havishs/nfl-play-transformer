"""
Builds a per-player, per-(season,week) table of CAUSAL rolling stats --
i.e. for a game in a given season/week, only stats from strictly earlier
season/weeks are used. This is what prevents the model from "seeing the
future" through a player's embedding.

Two source tables, since they're tracked separately by nflverse:
  - weekly_data       (offense: QB/RB/WR/TE, EPA-based)
  - weekly_pfr('def') (defense: DB/LB/DL, coverage + pass-rush)
There is no public per-play OL performance data -- OL falls back to
bio/usage features only (see BIO_FEATS).
"""

import numpy as np
import pandas as pd

OFF_FEATS = ["passing_epa", "rushing_epa", "receiving_epa", "target_share"]
DEF_FEATS = ["def_completion_pct", "def_passer_rating_allowed", "def_pressures",
             "def_sacks", "def_missed_tackle_pct"]
BIO_FEATS = ["years_exp", "draft_number"]  # available for every position, incl. OL


def _order_key(season, week):
    """Chronological sort key across season boundaries. Safe as long as week < 100."""
    return season * 100 + week


def load_sources(seasons, data_dir="../data"):
    """Load and concatenate weekly offense/defense stats + roster/crosswalk across seasons."""
    off_frames, def_frames, roster_frames = [], [], []
    for s in seasons:
        off_frames.append(pd.read_parquet(f"{data_dir}/weekly_off_{s}.parquet"))
        def_frames.append(pd.read_parquet(f"{data_dir}/weekly_def_{s}.parquet"))
        roster_frames.append(pd.read_parquet(f"{data_dir}/roster_{s}.parquet"))
    weekly_off = pd.concat(off_frames, ignore_index=True)
    weekly_def = pd.concat(def_frames, ignore_index=True)
    roster = pd.concat(roster_frames, ignore_index=True).drop_duplicates("player_id", keep="last")

    crosswalk = roster[["player_id", "pfr_id", "player_name", "position",
                         "years_exp", "draft_number"]].drop_duplicates("player_id")
    pfr_to_gsis = crosswalk.dropna(subset=["pfr_id"]).set_index("pfr_id")["player_id"].to_dict()
    weekly_def = weekly_def.copy()
    weekly_def["player_id"] = weekly_def["pfr_player_id"].map(pfr_to_gsis)
    weekly_def = weekly_def.dropna(subset=["player_id"])

    weekly_off = weekly_off.copy()
    weekly_off["order_key"] = _order_key(weekly_off["season"], weekly_off["week"])
    weekly_def["order_key"] = _order_key(weekly_def["season"], weekly_def["week"])

    return weekly_off, weekly_def, crosswalk


class PlayerFeatureLookup:
    """
    Precomputes causal cumulative-mean stats per player at every (season, week)
    they appear, so lookups during dataset assembly are O(1) instead of
    re-filtering the whole table per play.
    """

    def __init__(self, seasons, data_dir="../data"):
        self.weekly_off, self.weekly_def, self.crosswalk = load_sources(seasons, data_dir)
        self.bio = self.crosswalk.set_index("player_id")[["position", "years_exp", "draft_number"]]
        self._off_cache = self._build_cumulative(self.weekly_off, OFF_FEATS)
        self._def_cache = self._build_cumulative(self.weekly_def, DEF_FEATS)

    @staticmethod
    def _build_cumulative(table, feats):
        """
        For each player, at each order_key they have a row, compute the mean of
        `feats` over all STRICTLY EARLIER rows. Returns dict: player_id -> DataFrame
        indexed by order_key, columns = feats, plus 'n_prior_games'.
        """
        out = {}
        for pid, g in table.sort_values("order_key").groupby("player_id"):
            g = g.set_index("order_key")
            # shift(1).expanding().mean() = "mean of everything strictly before this row"
            cum = g[feats].expanding().mean().shift(1)
            cum["n_prior_games"] = np.arange(len(g))  # count of prior games at each row
            out[pid] = cum
        return out

    def lookup(self, player_id, season, week):
        """
        Returns (feature_dict, has_stats_flag) for a player going into a game
        at (season, week). Falls back to bio-only if no prior game stats exist
        (true for OL always, and for anyone before their first tracked game).
        """
        key = _order_key(season, week)
        pos = self.bio["position"].get(player_id, "UNK")
        years_exp = self.bio["years_exp"].get(player_id, np.nan)
        draft_number = self.bio["draft_number"].get(player_id, np.nan)
        bio = {"years_exp": years_exp, "draft_number": draft_number}

        off_row, def_row = None, None
        if player_id in self._off_cache:
            tbl = self._off_cache[player_id]
            prior = tbl[tbl.index < key]
            if len(prior):
                off_row = prior.iloc[-1]
        if player_id in self._def_cache:
            tbl = self._def_cache[player_id]
            prior = tbl[tbl.index < key]
            if len(prior):
                def_row = prior.iloc[-1]

        has_off = off_row is not None and not off_row[OFF_FEATS].isna().all()
        has_def = def_row is not None and not def_row[DEF_FEATS].isna().all()

        off_feats = {f: (float(off_row[f]) if has_off and not np.isnan(off_row[f]) else 0.0) for f in OFF_FEATS}
        def_feats = {f: (float(def_row[f]) if has_def and not np.isnan(def_row[f]) else 0.0) for f in DEF_FEATS}

        return {
            "position": pos,
            "bio": bio,
            "off_feats": off_feats,
            "def_feats": def_feats,
            "has_off_stats": bool(has_off),
            "has_def_stats": bool(has_def),
        }
