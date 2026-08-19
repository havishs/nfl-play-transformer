"""
Monte Carlo win-probability evaluation harness.

For a sample of held-out validation games, at the real start of each
quarter, runs many seeded rollouts through GameSimulator and tallies a win
probability, then compares it against the real final score. See
docs/superpowers/specs/2026-08-19-monte-carlo-win-probability-design.md for
the full design and the compute-budget reasoning behind the constants below.
"""

import hashlib
import random

import pandas as pd
import torch

import team_form as tf
from dataset import PlayDataset
from generate import GameSimulator, GameState
from get_batch import build_game_index
from model import GameTransformer
from special_teams_features import SpecialTeamsFeatureLookup
from train import (
    CACHE_PATH, CHECKPOINT_PATH, DATA_DIR, DEVICE, HISTORY_SEASONS, SEED,
    TRAINING_SEASONS, VAL_FRACTION,
)

QUARTERS = [1, 2, 3, 4]
N_EVAL_GAMES = 25
ROLLOUTS_PER_STATE = 500
MAX_PLAYS = 400
RESULTS_CSV_PATH = "win_prob_results.csv"


def real_state_at_quarter_start(game_df, quarter):
    """
    First real scrimmage play (down not null) in `quarter`, for one game's
    raw pbp rows. `game_df` must already be sorted by play_id. Returns a
    GameState with play_in_quarter=0 (this harness's own definition of
    "start of quarter"), or None if the quarter has no scrimmage play at all
    (e.g. a game that ran out the clock without a real 4th-quarter snap).
    """
    rows = game_df[(game_df["qtr"] == quarter) & game_df["down"].notna()]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return GameState(
        quarter=quarter, play_in_quarter=0,
        down=int(row["down"]), ydstogo=int(row["ydstogo"]),
        yardline_100=int(row["yardline_100"]),
        posteam=row["posteam"], defteam=row["defteam"],
        posteam_score=int(row["posteam_score"]), defteam_score=int(row["defteam_score"]),
    )


def real_outcome(game_df, team_a):
    """1.0 if team_a actually won this real game, 0.0 if it lost, 0.5 on a real tie."""
    row = game_df.iloc[0]
    team_a_score = row["home_score"] if team_a == row["home_team"] else row["away_score"]
    team_b_score = row["away_score"] if team_a == row["home_team"] else row["home_score"]
    if team_a_score > team_b_score:
        return 1.0
    if team_a_score < team_b_score:
        return 0.0
    return 0.5


def rollout_outcome(sim, log, initial_state):
    """
    "team_a", "team_b", or "unresolved" (MAX_PLAYS exhausted mid-overtime,
    still tied -- counted as 0.5/0.5 credit by win_probability).

    "Unresolved" is detected via a proxy -- quarter > 4 and the score still
    tied -- rather than a direct signal that generate()'s play loop ran out
    of plays without its own score-differs-in-OT stopping condition firing.
    This proxy is correct given this harness's constants (MAX_PLAYS=400 is
    generous relative to generate.py's PLAYS_PER_QUARTER=~43), but if those
    constants change, re-examine whether the proxy still holds.
    """
    final = log[-1]["state"] if log else initial_state
    if final.quarter > 4 and final.posteam_score == final.defteam_score:
        return "unresolved"
    if final.posteam == sim.team_a:
        team_a_score, team_b_score = final.posteam_score, final.defteam_score
    else:
        team_a_score, team_b_score = final.defteam_score, final.posteam_score
    return "team_a" if team_a_score > team_b_score else "team_b"


def state_seed(game_id, quarter):
    """
    Stable (process-independent) integer seed for one (game, quarter) starting
    point, used as the base seed for that state's rollouts (rollout i uses
    base_seed + i). Uses hashlib rather than Python's built-in hash(), which
    is randomized per-process and would break reproducibility across runs.
    """
    digest = hashlib.sha256(f"{game_id}:{quarter}".encode()).hexdigest()
    return int(digest[:8], 16)


def win_probability(sim, initial_state, n_rollouts, base_seed, max_plays):
    wins_a = 0.0
    for i in range(n_rollouts):
        sim.form_state = tf.initial_team_form()
        generator = torch.Generator().manual_seed(base_seed + i)
        log = sim.generate(max_plays, initial_state, generator=generator)
        outcome = rollout_outcome(sim, log, initial_state)
        if outcome == "team_a":
            wins_a += 1.0
        elif outcome == "unresolved":
            wins_a += 0.5
    return wins_a / n_rollouts


def sample_validation_games(dataset, n_games, seed):
    """
    Deterministic subsample of the held-out validation split (same split
    eval.py uses: the last VAL_FRACTION of games, chronologically). See the
    design doc for why this harness subsamples rather than evaluating the
    full validation set (full-set compute is not tractable given generate()
    isn't batched across rollouts).
    """
    game_ids = sorted(build_game_index(dataset.examples).keys())
    n_val = max(1, round(len(game_ids) * VAL_FRACTION))
    val_game_ids = game_ids[-n_val:]
    rng = random.Random(seed)
    return sorted(rng.sample(val_game_ids, min(n_games, len(val_game_ids))))


def _is_correct(p_hat, y):
    if p_hat == 0.5:
        return False
    predicted_a_wins = p_hat > 0.5
    actual_a_wins = y > 0.5
    return predicted_a_wins == actual_a_wins


def _metrics(records):
    if not records:
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan")}
    correct = sum(1 for p_hat, y in records if _is_correct(p_hat, y))
    brier = sum((p_hat - y) ** 2 for p_hat, y in records) / len(records)
    return {"n": len(records), "accuracy": correct / len(records), "brier": brier}


def summarize(records_by_quarter):
    """records_by_quarter: dict quarter -> list of (p_hat, y). Adds an "overall" key pooling every quarter."""
    summary = {quarter: _metrics(records) for quarter, records in records_by_quarter.items()}
    all_records = [r for records in records_by_quarter.values() for r in records]
    summary["overall"] = _metrics(all_records)
    return summary


def load_pbp_for_seasons(seasons, data_dir):
    frames = [pd.read_parquet(f"{data_dir}/pbp_{s}.parquet") for s in seasons]
    return pd.concat(frames, ignore_index=True)


def game_rows(pbp, game_id):
    return pbp[pbp["game_id"] == game_id].sort_values("play_id")


def evaluate(model, dataset, pbp, game_ids, device, special_teams=None,
             rollouts_per_state=ROLLOUTS_PER_STATE, max_plays=MAX_PLAYS):
    """
    Runs the full harness over `game_ids`. Returns (records_by_quarter,
    csv_rows, skipped): records_by_quarter maps quarter -> list of (p_hat, y)
    for summarize(); csv_rows is the flat per-state log for the CSV dump;
    skipped counts game-quarter points with no real scrimmage play that
    quarter (see real_state_at_quarter_start).
    """
    model.eval()

    records_by_quarter = {q: [] for q in QUARTERS}
    csv_rows = []
    skipped = 0

    for game_id in game_ids:
        game_df = game_rows(pbp, game_id)
        sim = GameSimulator(model, dataset, game_id, device=device, special_teams=special_teams)
        y = real_outcome(game_df, sim.team_a)

        for quarter in QUARTERS:
            initial_state = real_state_at_quarter_start(game_df, quarter)
            if initial_state is None:
                skipped += 1
                continue
            base_seed = state_seed(game_id, quarter)
            p_hat = win_probability(sim, initial_state, rollouts_per_state, base_seed, max_plays)
            records_by_quarter[quarter].append((p_hat, y))
            csv_rows.append({"game_id": game_id, "quarter": quarter, "p_hat": p_hat, "y": y})

    return records_by_quarter, csv_rows, skipped


def main():
    print(f"device: {DEVICE}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print(f"loaded checkpoint from step {checkpoint['step']}, val_loss {checkpoint['val_loss']:.4f}")

    print(f"building dataset (cache: {CACHE_PATH})...")
    dataset = PlayDataset(HISTORY_SEASONS, TRAINING_SEASONS, DATA_DIR, cache_path=CACHE_PATH)

    model = GameTransformer(checkpoint["vocabs"], **checkpoint["hyperparameters"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"loading raw pbp for seasons {TRAINING_SEASONS}...")
    pbp = load_pbp_for_seasons(TRAINING_SEASONS, DATA_DIR)

    print("loading special-teams (kicker/punter) data...")
    special_teams = SpecialTeamsFeatureLookup(TRAINING_SEASONS, DATA_DIR)

    game_ids = sample_validation_games(dataset, N_EVAL_GAMES, SEED)
    print(f"evaluating {len(game_ids)} validation games x {len(QUARTERS)} in-game points x "
          f"{ROLLOUTS_PER_STATE} rollouts each...")

    records_by_quarter, csv_rows, skipped = evaluate(
        model, dataset, pbp, game_ids, DEVICE, special_teams=special_teams,
    )
    print(f"skipped {skipped} game-quarter points with no real scrimmage play that quarter")

    summary = summarize(records_by_quarter)
    print("\n--- Monte Carlo win-probability evaluation ---")
    print(f"{'quarter':10s}{'n':>6s}{'accuracy':>12s}{'brier':>10s}")
    for quarter in QUARTERS:
        m = summary[quarter]
        print(f"Q{quarter:<9d}{m['n']:>6d}{m['accuracy']*100:>11.1f}%{m['brier']:>10.4f}")
    m = summary["overall"]
    print(f"{'overall':10s}{m['n']:>6d}{m['accuracy']*100:>11.1f}%{m['brier']:>10.4f}")

    pd.DataFrame(csv_rows).to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nraw per-state results written to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
