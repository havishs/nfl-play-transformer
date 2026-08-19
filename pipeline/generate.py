"""
generate(): minimal autoregressive rollout for model v0.

Scope, per an explicit decision (this is bigger than everything else in v0
combined, so it was scoped deliberately rather than guessed): SCRIMMAGE
PLAYS ONLY. The model's 4 output heads (yards_gained/touchdown/turnover/
return_yards) predict a scrimmage play's outcome, applied to real down/
distance/field-position/score transition logic.

4th-down decisions, field goals, punts, and extra points now have real
mechanics -- a 3-way decision (kick/punt/go-for-it) grounded in real
2018-2023 pbp distributions, an empirical distance-based FG success model
adjusted by each team's actual causal kicker FG%, a per-punter causal
distance model, a real (non-100%) XP success rate, turnover-on-downs for
a failed go-for-it, and sudden-death overtime -- see
docs/superpowers/specs/2026-08-19-special-teams-modeling-design.md for the
full design and all the real-data grounding. Still genuinely simplified:
kickoffs are always a touchback (no returns), blocked kicks/punts and
2-point conversions aren't modeled, and personnel (including which player
is on the field) stays fixed for the whole rollout -- no substitutions, no
in-game roster changes.

Two things the model can't do (yet), worked around here:
  - play_type isn't predicted (model.py explains why: it's currently only
    a situational input, predicting it back out would leak). Each step's
    play_type input is chosen by a simple weighted-random heuristic, not
    the model.
  - Personnel: design points 1 (learned history encoder) and 2 (in-game
    running form) are still stubbed. This reuses one real game's opening
    personnel (its first play's offense/defense) as a FIXED "who's on the
    field" roster for both teams for the whole rollout -- no substitution
    modeling, no in-game form evolution.

Predicted outcomes are class buckets (e.g. "short_1-3"), not exact yards --
the model only ever learned to predict buckets. Applying a bucket to a
concrete game state needs a real number, so predictions are mapped to each
bucket's midpoint (BUCKET_MIDPOINT below, derived from situational.py's own
edges, not invented). This is a genuine (documented) simplification.
"""

import random
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

import situational as sit
import team_form as tf
from special_teams_features import SpecialTeamsFeatureLookup

TOUCHBACK_YARDLINE_100 = 75  # own 25, kickoff touchback spot
PUNT_TOUCHBACK_YARDLINE_100 = 80  # own 20, punt touchback spot (real NFL rule -- distinct from kickoff's)
PUNT_TOUCHBACK_RATE = 0.067  # real 2018-2023 rate
PUNT_AVG_RETURN_YARDS = 3.6  # real 2018-2023 mean return yardage when not a touchback
LEAGUE_AVG_PUNT_DISTANCE = 45.9  # real 2018-2023 mean gross kick_distance -- fallback for an unknown punter
PLAYS_PER_QUARTER = 43  # ~174 measured plays/game (PROJECT_BRIEF.md) / 4 quarters
PASS_RUN_WEIGHTS = {"pass": 0.57, "run": 0.43}  # rough league split, only used as model INPUT context

# 4th-down decision + field goal constants -- all derived from real 2018-2023
# pbp data, see docs/superpowers/specs/2026-08-19-special-teams-modeling-design.md
FG_RANGE_YARDLINE_100 = 40  # ~58-yard attempt; real attempts: p95=37, p99=41, max=52
GO_FOR_IT_SHORT_YDSTOGO = 2  # short-yardage conversions generally worth attempting
DESPERATION_SCORE_DEFICIT = 9  # down more than one converted score late in Q4 -- always go for it
KICK_DISTANCE_OFFSET = 18  # kick_distance = yardline_100 + 18 (real median/mode offset, exact)
FG_DISTANCE_EDGES = [19, 29, 39, 49, 59]  # upper bound of each bucket except the last (60+)
FG_DISTANCE_BASELINE = [0.98, 0.98, 0.93, 0.78, 0.67, 0.37]  # real make rate by bucket, 2018-2023
LEAGUE_AVG_FG_PCT = 0.846  # overall real make rate, 2018-2023 -- kicker-adjustment baseline

YARDS_GAINED_BUCKET_MIDPOINT = {
    "loss": -3, "no_gain": 0, "short_1-3": 2, "medium_4-6": 5,
    "good_7-10": 8, "big_11-20": 15, "explosive_20+": 25,
}
RETURN_YARDS_BUCKET_MIDPOINT = {
    "none": 0, "short_1-5": 3, "medium_6-15": 10, "long_16-30": 22, "explosive_30+": 35,
}


@dataclass(frozen=True)
class GameState:
    quarter: int
    play_in_quarter: int
    down: int
    ydstogo: int
    yardline_100: int  # distance to the endzone the CURRENT posteam is driving toward
    posteam: str
    defteam: str
    posteam_score: int
    defteam_score: int

    def flip_possession(self, **overrides):
        base = replace(
            self, posteam=self.defteam, defteam=self.posteam,
            posteam_score=self.defteam_score, defteam_score=self.posteam_score,
        )
        return replace(base, **overrides)


def _time_bucket_for(play_in_quarter):
    frac_remaining = max(0.0, 1 - play_in_quarter / PLAYS_PER_QUARTER)
    if frac_remaining > 2 / 3:
        return ">10min"
    if frac_remaining > 1 / 3:
        return "5-10min"
    if frac_remaining > 2 / 15:
        return "2-5min"
    return "<2min"


def _situational_dict(state, play_type):
    down_b = sit.bucket_down(float(state.down))
    return {
        "play_type": play_type,
        "down": down_b,
        "distance": sit.bucket_distance(state.ydstogo, down_b),
        "field_zone": sit.bucket_field_zone(state.yardline_100),
        "score_diff": sit.bucket_score_diff(state.posteam_score - state.defteam_score),
        "quarter": sit.bucket_quarter(float(state.quarter)),
        "time_bucket": _time_bucket_for(state.play_in_quarter),
        "posteam": state.posteam,
        "defteam": state.defteam,
    }


def _sample_class(logits, generator):
    # sample on CPU regardless of the model's device -- torch.multinomial requires
    # the generator's device to match the input tensor's, and generator here is
    # always a plain CPU generator (see GameSimulator.generate's docstring/usage).
    probs = F.softmax(logits, dim=-1).to("cpu")
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


def _rand(generator):
    # uniform float in [0, 1) -- CPU-only, same generator convention as _sample_class.
    return torch.rand(1, generator=generator).item()


def _weighted_choice(weights, generator):
    keys = list(weights)
    values = torch.tensor(list(weights.values()))
    idx = torch.multinomial(values, num_samples=1, generator=generator).item()
    return keys[idx]


def _punt(state, punter_avg_distance):
    """punter_avg_distance: causal career average gross kick_distance for the punting team's punter, or None if unknown (falls back to the league average)."""
    if random.random() < PUNT_TOUCHBACK_RATE:
        return state.flip_possession(
            down=1, ydstogo=10, yardline_100=PUNT_TOUCHBACK_YARDLINE_100,
            play_in_quarter=state.play_in_quarter + 1,
        )
    gross_distance = punter_avg_distance if punter_avg_distance is not None else LEAGUE_AVG_PUNT_DISTANCE
    net_distance = gross_distance - PUNT_AVG_RETURN_YARDS
    kicking_yardline_100 = max(1, state.yardline_100 - net_distance)
    receiving_yardline_100 = int(round(min(PUNT_TOUCHBACK_YARDLINE_100, 100 - kicking_yardline_100)))
    return state.flip_possession(
        down=1, ydstogo=10, yardline_100=receiving_yardline_100,
        play_in_quarter=state.play_in_quarter + 1,
    )


def _fourth_down_decision(state):
    """Returns "fg", "punt", or "go" -- see the design doc for the reasoning behind each threshold."""
    desperate = state.quarter == 4 and (state.posteam_score - state.defteam_score) <= -DESPERATION_SCORE_DEFICIT
    if desperate:
        return "go"
    if state.yardline_100 <= FG_RANGE_YARDLINE_100:
        return "fg"
    if state.ydstogo <= GO_FOR_IT_SHORT_YDSTOGO:
        return "go"
    return "punt"


def _fg_make_probability(kick_distance, kicker_fg_pct):
    idx = 0
    for edge in FG_DISTANCE_EDGES:
        if kick_distance > edge:
            idx += 1
        else:
            break
    baseline = FG_DISTANCE_BASELINE[idx]
    adjustment = (kicker_fg_pct - LEAGUE_AVG_FG_PCT) if kicker_fg_pct is not None else 0.0
    return min(0.99, max(0.02, baseline + adjustment))


def _attempt_field_goal(state, kicker_fg_pct):
    """kicker_fg_pct: causal career FG% for the kicking team's kicker, or None if unknown (falls back to the distance baseline)."""
    kick_distance = state.yardline_100 + KICK_DISTANCE_OFFSET
    p_make = _fg_make_probability(kick_distance, kicker_fg_pct)
    made = random.random() < p_make
    if made:
        scored = replace(state, posteam_score=state.posteam_score + 3)
        return _kickoff_after_score(scored), "field_goal_made"
    # missed: opponent takes over at the previous line of scrimmage (mirrors
    # _apply_turnover's math with yards_gained=0, return_yards=0)
    new_state = state.flip_possession(
        down=1, ydstogo=10, yardline_100=100 - state.yardline_100,
        play_in_quarter=state.play_in_quarter + 1,
    )
    return new_state, "field_goal_missed"


def _kickoff_after_score(state):
    """Scoring-team-kicks-off heuristic: always a touchback."""
    return state.flip_possession(
        down=1, ydstogo=10, yardline_100=TOUCHBACK_YARDLINE_100,
        play_in_quarter=state.play_in_quarter + 1,
    )


XP_SUCCESS_RATE = 0.942  # real 2018-2023 XP make rate -- fixed since XP distance doesn't vary by kicker


def _attempt_extra_point(state):
    made = random.random() < XP_SUCCESS_RATE
    return replace(state, posteam_score=state.posteam_score + (1 if made else 0))


def _apply_touchdown(state):
    scored = replace(state, posteam_score=state.posteam_score + 6)
    scored = _attempt_extra_point(scored)  # no 2pt decision modeled, matches existing simplification
    return _kickoff_after_score(scored)


def _apply_turnover(state, yards_gained, return_yards):
    """
    Ball spot from the NEW posteam's perspective: the play gains yards_gained
    toward the OLD posteam's opponent goal (yardline_100 decreases), then the
    return runs the other way, back toward the old posteam's own goal
    (yardline_100 increases again from the old posteam's perspective).
    """
    spot_from_old_posteam = state.yardline_100 - yards_gained + return_yards
    new_yardline_100 = min(99, max(1, 100 - spot_from_old_posteam))
    return state.flip_possession(
        down=1, ydstogo=10, yardline_100=new_yardline_100,
        play_in_quarter=state.play_in_quarter + 1,
    )


def _apply_turnover_on_downs(state, yards_gained):
    """4th down, didn't convert: possession flips at the spot, no return (mirrors _apply_turnover's math with return_yards=0)."""
    spot_from_old_posteam = state.yardline_100 - yards_gained
    new_yardline_100 = min(99, max(1, 100 - spot_from_old_posteam))
    return state.flip_possession(
        down=1, ydstogo=10, yardline_100=new_yardline_100,
        play_in_quarter=state.play_in_quarter + 1,
    )


def _normalize_quarter(state):
    if state.play_in_quarter >= PLAYS_PER_QUARTER:
        return replace(state, quarter=state.quarter + 1, play_in_quarter=0)
    return state


def _apply_scrimmage_gain(state, yards_gained):
    new_yardline_100 = min(100, max(1, state.yardline_100 - yards_gained))  # clipped: touchdown is handled separately
    new_ydstogo = state.ydstogo - yards_gained
    if new_ydstogo <= 0:
        down, ydstogo = 1, min(10, new_yardline_100)
    else:
        down, ydstogo = state.down + 1, new_ydstogo
    return replace(
        state, down=down, ydstogo=ydstogo, yardline_100=new_yardline_100,
        play_in_quarter=state.play_in_quarter + 1,
    )


class GameSimulator:
    def __init__(self, model, dataset, seed_game_id, device="cpu", special_teams=None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.block_size = model.block_size

        start = next(i for i, ex in enumerate(dataset.examples) if ex["game_id"] == seed_game_id)
        seed = dataset.examples[start]
        self.season = int(seed_game_id.split("_")[0])
        self.week = seed["week"]
        self.team_a = seed["situational"]["posteam"]
        self.team_a_personnel = (seed["offense"], seed["defense"])
        self.team_b = seed["situational"]["defteam"]
        self.team_b_personnel = (seed["defense"], seed["offense"])
        self.form_state = tf.initial_team_form()

        # special_teams=None (e.g. in small/fast unit tests) leaves every stat
        # at None, which already falls back correctly to the league-average
        # baseline everywhere it's used (see _fg_make_probability/_punt).
        self.team_a_fg_pct = self.team_b_fg_pct = None
        self.team_a_punt_avg = self.team_b_punt_avg = None
        if special_teams is not None:
            for team_attr, fg_attr, punt_attr in [
                (self.team_a, "team_a_fg_pct", "team_a_punt_avg"),
                (self.team_b, "team_b_fg_pct", "team_b_punt_avg"),
            ]:
                kicker_id = special_teams.primary_kicker(team_attr, self.season)
                if kicker_id is not None:
                    setattr(self, fg_attr, special_teams.fg_pct(kicker_id, self.season, self.week))
                punter_id = special_teams.primary_punter(team_attr, self.season)
                if punter_id is not None:
                    setattr(self, punt_attr, special_teams.punt_avg_distance(punter_id, self.season, self.week))

    def _fg_pct_for(self, posteam):
        return self.team_a_fg_pct if posteam == self.team_a else self.team_b_fg_pct

    def _punt_avg_for(self, posteam):
        return self.team_a_punt_avg if posteam == self.team_a else self.team_b_punt_avg

    def _personnel_for(self, posteam):
        # Personnel is a fixed snapshot (no substitution/in-game-form modeling, per
        # the design doc) -- history is looked up once, at the seed game's cutoff,
        # and held constant for the whole rollout, consistent with that.
        offense, defense = self.team_a_personnel if posteam == self.team_a else self.team_b_personnel
        return (
            self.dataset._personnel_tensors(offense),
            self.dataset._personnel_tensors(defense),
            self.dataset._history_tensors(offense, self.season, self.week),
            self.dataset._history_tensors(defense, self.season, self.week),
        )

    def _tensorize_step(self, state, play_type):
        situational = _situational_dict(state, play_type)
        (off_pos, off_feat), (def_pos, def_feat), (off_hist, off_hist_mask), (def_hist, def_hist_mask) = \
            self._personnel_for(state.posteam)
        return {
            "situational": self.dataset._situational_tensor(situational),
            "offense_position": off_pos,
            "offense_features": off_feat,
            "offense_history": off_hist,
            "offense_history_mask": off_hist_mask,
            "defense_position": def_pos,
            "defense_features": def_feat,
            "defense_history": def_hist,
            "defense_history_mask": def_hist_mask,
            "team_form": torch.tensor(
                tf.team_form_features(self.form_state, state.posteam, state.defteam), dtype=torch.float32
            ),
        }

    def generate(self, n_plays, initial_state, generator=None):
        state = initial_state
        window = []
        log = []

        for _ in range(n_plays):
            if state.quarter > 4 and state.posteam_score != state.defteam_score:
                break

            if state.down > 3:
                decision = _fourth_down_decision(state)
                if decision == "punt":
                    state = _normalize_quarter(_punt(state, self._punt_avg_for(state.posteam)))
                    log.append({"event": "punt", "state": state})
                    continue
                if decision == "fg":
                    state, fg_event = _attempt_field_goal(state, self._fg_pct_for(state.posteam))
                    state = _normalize_quarter(state)
                    log.append({"event": fg_event, "state": state})
                    continue
                # decision == "go": fall through to the normal model-driven
                # scrimmage-play flow below, same as any other down. A
                # non-converting result is handled by the
                # turnover_on_downs branch further down in the outcome
                # dispatch (state.down == 4 and it wasn't a score/turnover).

            play_type = random.choices(
                list(PASS_RUN_WEIGHTS), weights=list(PASS_RUN_WEIGHTS.values())
            )[0]
            step = self._tensorize_step(state, play_type)
            window.append(step)
            window = window[-self.block_size:]

            batch = {
                k: torch.stack([w[k] for w in window]).unsqueeze(0).to(self.device)
                for k in window[0].keys()
            }
            with torch.no_grad():
                logits, _ = self.model(batch)

            last = -1
            yards_bucket = self.dataset.vocabs["yards_gained"]
            return_bucket = self.dataset.vocabs["return_yards"]
            yards_idx = _sample_class(logits["yards_gained"][0, last], generator)
            touchdown = bool(_sample_class(logits["touchdown"][0, last], generator))
            turnover = bool(_sample_class(logits["turnover"][0, last], generator))
            return_idx = _sample_class(logits["return_yards"][0, last], generator)

            yards_label = [k for k, v in yards_bucket.items() if v == yards_idx][0]
            return_label = [k for k, v in return_bucket.items() if v == return_idx][0]
            yards_gained = YARDS_GAINED_BUCKET_MIDPOINT[yards_label]
            return_yards = RETURN_YARDS_BUCKET_MIDPOINT[return_label]

            # state.posteam/state.defteam here are still PRE-play (state isn't
            # reassigned until the branch below) -- same causal "capture
            # current, then advance" order build_dataset.py uses.
            self.form_state = tf.update_team_form(
                self.form_state, state.posteam, state.defteam, yards_gained, True, touchdown, turnover, True,
            )

            if turnover:
                state = _apply_turnover(state, yards_gained, return_yards)
                event = "turnover"
            elif touchdown:
                state = _apply_touchdown(state)
                event = "touchdown"
            elif state.down == 4 and (state.ydstogo - yards_gained) > 0:
                state = _apply_turnover_on_downs(state, yards_gained)
                event = "turnover_on_downs"
            else:
                state = _apply_scrimmage_gain(state, yards_gained)
                event = "gain"

            state = _normalize_quarter(state)
            log.append({"event": event, "play_type": play_type, "yards_gained": yards_gained, "state": state})

        return log
