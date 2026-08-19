"""
Team-level in-game form: causal, per-game EMA over each team's own recent
play outcomes (offense) and outcomes allowed (defense). Unlike
player_features.py's career-long stats, this resets every game and updates
live -- shared between build_dataset.py (training, real outcomes) and
generate.py (rollout, the model's own sampled outcomes), via the same
update function. See docs/superpowers/specs/2026-08-18-in-game-team-form-design.md.
"""

import numpy as np
import pandas as pd

EMA_ALPHA = 0.25  # recency weight per update -- higher = more reactive to recent plays


def _empty_side():
    return {
        "yards_ema": 0.0,
        "touchdown_ema": 0.0,
        "turnover_ema": 0.0,
        "has_yards_history": False,
        "has_td_turnover_history": False,
    }


def initial_team_form():
    """Fresh state for the start of a game/rollout -- teams are added lazily on first update."""
    return {}


def _update_side(side, yards_gained, yards_applicable, touchdown, turnover, td_turnover_applicable):
    side = dict(side)
    if yards_applicable:
        if side["has_yards_history"]:
            side["yards_ema"] = EMA_ALPHA * yards_gained + (1 - EMA_ALPHA) * side["yards_ema"]
        else:
            side["yards_ema"] = float(yards_gained)
        side["has_yards_history"] = True
    if td_turnover_applicable:
        td_val, to_val = float(touchdown), float(turnover)
        if side["has_td_turnover_history"]:
            side["touchdown_ema"] = EMA_ALPHA * td_val + (1 - EMA_ALPHA) * side["touchdown_ema"]
            side["turnover_ema"] = EMA_ALPHA * to_val + (1 - EMA_ALPHA) * side["turnover_ema"]
        else:
            side["touchdown_ema"] = td_val
            side["turnover_ema"] = to_val
        side["has_td_turnover_history"] = True
    return side


def update_team_form(form_state, posteam, defteam, yards_gained, yards_applicable,
                      touchdown, turnover, td_turnover_applicable):
    """
    Non-mutating: returns a new form_state with posteam's "offense" side and
    defteam's "defense" side updated from this play's outcome. Call this
    AFTER capturing this play's own team_form_features() -- the update
    reflects what's known starting from the NEXT play, not this one.

    posteam/defteam are only validated (non-null, distinct) when the play
    is actually applicable (yards_applicable or td_turnover_applicable) --
    real pbp data has some non-applicable rows with missing team codes
    (e.g. certain no_play rows), and those rows never touch team state
    either way, so there's nothing to validate for them.
    """
    if yards_applicable or td_turnover_applicable:
        assert pd.notna(posteam) and pd.notna(defteam), \
            f"posteam and defteam must both be real values on an applicable play, got posteam={posteam!r} defteam={defteam!r}"
        assert posteam != defteam, f"posteam and defteam must differ, got {posteam!r} for both"
    new_state = dict(form_state)
    posteam_sides = dict(new_state.get(posteam, {"offense": _empty_side(), "defense": _empty_side()}))
    defteam_sides = dict(new_state.get(defteam, {"offense": _empty_side(), "defense": _empty_side()}))

    posteam_sides["offense"] = _update_side(
        posteam_sides["offense"], yards_gained, yards_applicable, touchdown, turnover, td_turnover_applicable
    )
    defteam_sides["defense"] = _update_side(
        defteam_sides["defense"], yards_gained, yards_applicable, touchdown, turnover, td_turnover_applicable
    )

    new_state[posteam] = posteam_sides
    new_state[defteam] = defteam_sides
    return new_state


def _side_vector(side):
    return [
        side["yards_ema"], side["touchdown_ema"], side["turnover_ema"],
        float(side["has_yards_history"]), float(side["has_td_turnover_history"]),
    ]


def team_form_features(form_state, posteam, defteam):
    """Play-level feature vector: posteam's offense side (5) ++ defteam's defense side (5) -> (10,)."""
    posteam_offense = form_state.get(posteam, {}).get("offense", _empty_side())
    defteam_defense = form_state.get(defteam, {}).get("defense", _empty_side())
    return np.array(_side_vector(posteam_offense) + _side_vector(defteam_defense), dtype=np.float32)
