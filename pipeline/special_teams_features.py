"""
Causal per-kicker FG% and per-punter average gross punt distance. Unlike
player_features.py's weekly_off/weekly_def sources, this is sourced
directly from pbp's own field_goal_result/kicker_player_id/kick_distance
and punter_player_id/kick_distance columns -- no separate data pull
needed, that data already exists in the same parquet files used
throughout this pipeline. Same causal "strictly earlier (season, week)
only" cutoff as PlayerFeatureLookup. See docs/superpowers/specs/
2026-08-19-special-teams-modeling-design.md.
"""

import pandas as pd


def _order_key(season, week):
    """Chronological sort key across season boundaries. Safe as long as week < 100."""
    return season * 100 + week


def load_special_teams_plays(seasons, data_dir="../data"):
    """Load and concatenate raw pbp across seasons, split into FG and punt attempts."""
    frames = [pd.read_parquet(f"{data_dir}/pbp_{s}.parquet") for s in seasons]
    pbp = pd.concat(frames, ignore_index=True).copy()
    pbp["order_key"] = _order_key(pbp["season"], pbp["week"])

    fg = pbp[pbp["play_type"] == "field_goal"].copy()
    fg["made"] = (fg["field_goal_result"] == "made").astype(float)

    punt = pbp[pbp["play_type"] == "punt"].copy()

    return fg, punt


class SpecialTeamsFeatureLookup:
    """
    Caches raw per-attempt rows per kicker/punter, indexed by order_key (a
    kicker/punter can have several attempts in the same week, so this index
    isn't unique). Causal FG%/punt-distance is then the mean over whichever
    of those rows are strictly earlier than the (season, week) being looked
    up -- computed at lookup time rather than precomputed, since a running
    per-row expanding mean would leak same-week attempts into each other.
    """

    def __init__(self, seasons, data_dir="../data"):
        self.fg, self.punt = load_special_teams_plays(seasons, data_dir)
        self._fg_cache = self._build_raw(self.fg, "kicker_player_id", ["made"])
        self._punt_cache = self._build_raw(self.punt, "punter_player_id", ["kick_distance"])

    @staticmethod
    def _build_raw(table, id_col, feats):
        """Per-id DataFrame of each attempt's own (feats), indexed by order_key -- no aggregation."""
        out = {}
        for pid, g in table.sort_values("order_key").groupby(id_col):
            out[pid] = g.set_index("order_key")[feats]
        return out

    def fg_pct(self, kicker_player_id, season, week):
        """Causal career FG% as of (season, week). None if no prior real attempts."""
        key = _order_key(season, week)
        if kicker_player_id not in self._fg_cache:
            return None
        prior = self._fg_cache[kicker_player_id]
        prior = prior[prior.index < key]
        if not len(prior):
            return None
        value = prior["made"].mean()
        return None if pd.isna(value) else float(value)

    def punt_avg_distance(self, punter_player_id, season, week):
        """Causal career average gross kick_distance as of (season, week). None if no prior real attempts."""
        key = _order_key(season, week)
        if punter_player_id not in self._punt_cache:
            return None
        prior = self._punt_cache[punter_player_id]
        prior = prior[prior.index < key]
        if not len(prior):
            return None
        value = prior["kick_distance"].mean()
        return None if pd.isna(value) else float(value)

    def primary_kicker(self, team, season):
        """Most frequent real kicker_player_id for `team` in `season`'s own pbp. None if team has no FG attempts that season."""
        team_fg = self.fg[(self.fg["posteam"] == team) & (self.fg["season"] == season)]
        if not len(team_fg):
            return None
        return team_fg["kicker_player_id"].mode().iloc[0]

    def primary_punter(self, team, season):
        """Most frequent real punter_player_id for `team` in `season`'s own pbp. None if team has no punts that season."""
        team_punt = self.punt[(self.punt["posteam"] == team) & (self.punt["season"] == season)]
        if not len(team_punt):
            return None
        return team_punt["punter_player_id"].mode().iloc[0]
