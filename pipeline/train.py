"""
Training loop for GameTransformer (model v0), structured like the Karpathy
tutorial's own training loop (~/Downloads/gpt.py): hardcoded hyperparameters
at the top, periodic train/val loss reporting via estimate_loss().

Train/val split is by GAME, not by play (a play from the same game must
never appear in both splits). Uses a chronological holdout: game_id sorts
as "{season}_{week:02d}_{away}_{home}", so the last N games alphabetically
are the season's most recent weeks. Matches the project's causal ethos
elsewhere (PlayerFeatureLookup, bucket boundaries) -- validate on what the
model hasn't seen yet in time, not a random shuffle.
"""

from collections import Counter

import torch

from dataset import PlayDataset
from generate import GameSimulator, GameState
from get_batch import GameBatcher, build_game_index
from model import OUTPUT_HEADS, GameTransformer

# ---- hyperparameters ----
HISTORY_SEASONS = [2022]
TRAINING_SEASONS = [2023]
DATA_DIR = "../data"
VAL_FRACTION = 0.1
BLOCK_SIZE = 32
BATCH_SIZE = 16
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 20
LEARNING_RATE = 3e-4
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
CHECKPOINT_PATH = "checkpoint.pt"
# --------------------------


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = {tk: tv.to(device) for tk, tv in v.items()} if k == "targets" else v.to(device)
    return out


@torch.no_grad()
def estimate_loss(model, batcher, eval_iters, generator):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        batch = to_device(batcher.get_batch(BATCH_SIZE, generator=generator), DEVICE)
        _, loss = model(batch, batch["targets"])
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def _head_mask_name(name):
    return "td_turnover_mask" if name in ("touchdown", "turnover") else f"{name}_mask"


@torch.no_grad()
def estimate_accuracy(model, batcher, eval_iters, generator):
    """Masked top-1 accuracy per head -- only counts positions where that head's target is applicable."""
    model.eval()
    correct = {name: 0.0 for name in OUTPUT_HEADS}
    total = {name: 0.0 for name in OUTPUT_HEADS}
    for _ in range(eval_iters):
        batch = to_device(batcher.get_batch(BATCH_SIZE, generator=generator), DEVICE)
        logits, _ = model(batch, batch["targets"])
        for name in OUTPUT_HEADS:
            preds = logits[name].argmax(dim=-1)
            mask = batch["targets"][_head_mask_name(name)]
            correct[name] += ((preds == batch["targets"][name]).float() * mask).sum().item()
            total[name] += mask.sum().item()
    model.train()
    return {name: (correct[name] / total[name] if total[name] > 0 else float("nan")) for name in OUTPUT_HEADS}


def target_label_counts(dataset, game_ids):
    """Per-head label counts (only where that head's target is applicable), restricted to game_ids."""
    game_id_set = set(game_ids)
    counts = {name: Counter() for name in OUTPUT_HEADS}
    for ex in dataset.examples:
        if ex["game_id"] not in game_id_set:
            continue
        t = ex["targets"]
        if t["yards_gained_applicable"]:
            counts["yards_gained"][t["yards_gained"]] += 1
        if t["td_turnover_applicable"]:
            counts["touchdown"][t["touchdown"]] += 1
            counts["turnover"][t["turnover"]] += 1
        if t["return_yards_applicable"]:
            counts["return_yards"][t["return_yards"]] += 1
    return counts


def majority_baseline_accuracy(counts):
    """
    Simplest possible baseline for the masked-accuracy metric: always predict
    the single most common label. Gives context for whether the model's
    accuracy numbers mean anything -- e.g. if one bucket dominates >90% of
    plays, "high accuracy" there is trivial.
    """
    baseline = {}
    for name, c in counts.items():
        total = sum(c.values())
        baseline[name] = (max(c.values()) / total) if total else float("nan")
    return baseline


def main():
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    print(f"device: {DEVICE}")
    print("building dataset (full season, ~3 min)...")
    dataset = PlayDataset(HISTORY_SEASONS, TRAINING_SEASONS, DATA_DIR)

    game_ids = sorted(build_game_index(dataset.examples).keys())
    n_val = max(1, round(len(game_ids) * VAL_FRACTION))
    train_game_ids, val_game_ids = game_ids[:-n_val], game_ids[-n_val:]
    print(f"{len(train_game_ids)} train games, {len(val_game_ids)} val games (chronological holdout)")

    train_batcher = GameBatcher(dataset, BLOCK_SIZE, game_ids=train_game_ids)
    val_batcher = GameBatcher(dataset, BLOCK_SIZE, game_ids=val_game_ids)
    train_counts = target_label_counts(dataset, train_game_ids)

    model = GameTransformer(
        dataset.vocabs, block_size=BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, dropout=DROPOUT
    ).to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Best-checkpoint (early-stopping-by-checkpoint): a prior run trained to a
    # fixed step count and val loss diverged upward well before the end (train
    # loss kept falling, val loss rose -- classic overfitting on 257 training
    # games). Saving whenever val loss improves, rather than only at the final
    # step, means MAX_ITERS being too high costs wasted compute, not a worse
    # model. See the design doc addendum for the run that motivated this.
    best_val_loss = float("inf")
    best_step = None

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            train_loss = estimate_loss(model, train_batcher, EVAL_ITERS, generator)
            val_loss = estimate_loss(model, val_batcher, EVAL_ITERS, generator)
            print(f"step {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, best_step = val_loss, it
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocabs": dataset.vocabs,
                    "hyperparameters": {
                        "block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD,
                        "n_layer": N_LAYER, "dropout": DROPOUT,
                    },
                    "step": it, "val_loss": val_loss,
                }, CHECKPOINT_PATH)

        batch = to_device(train_batcher.get_batch(BATCH_SIZE, generator=generator), DEVICE)
        _, loss = model(batch, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\nbest checkpoint: step {best_step}, val loss {best_val_loss:.4f} (saved to {CHECKPOINT_PATH})")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=False)["model_state_dict"])

    print("\n--- val accuracy per head (masked top-1, vs. majority-class baseline from training games) ---")
    val_accuracy = estimate_accuracy(model, val_batcher, EVAL_ITERS * 3, generator)
    baseline = majority_baseline_accuracy(train_counts)
    for name in OUTPUT_HEADS:
        print(f"{name:15s} model={val_accuracy[name] * 100:5.1f}%   majority-baseline={baseline[name] * 100:5.1f}%")

    print("\n--- sample rollout from the best checkpoint (step 4: does generate() look plausible?) ---")
    model.eval()
    sim = GameSimulator(model, dataset, seed_game_id=val_game_ids[0], device=DEVICE)
    initial = GameState(
        quarter=1, play_in_quarter=0, down=1, ydstogo=10, yardline_100=75,
        posteam=sim.team_a, defteam=sim.team_b, posteam_score=0, defteam_score=0,
    )
    log = sim.generate(30, initial, generator=generator)
    for entry in log:
        s = entry["state"]
        print(f"{entry['event']:10s} q{s.quarter} down{s.down} ydstogo{s.ydstogo:3d} "
              f"yl100{s.yardline_100:3d}  {s.posteam} {s.posteam_score}-{s.defteam_score} {s.defteam}")


if __name__ == "__main__":
    main()
