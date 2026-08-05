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

import torch

from dataset import PlayDataset
from generate import GameSimulator, GameState
from get_batch import GameBatcher, build_game_index
from model import GameTransformer

# ---- hyperparameters ----
HISTORY_SEASONS = [2022]
TRAINING_SEASONS = [2023]
DATA_DIR = "../data"
VAL_FRACTION = 0.1
BLOCK_SIZE = 32
BATCH_SIZE = 16
MAX_ITERS = 500
EVAL_INTERVAL = 100
EVAL_ITERS = 20
LEARNING_RATE = 3e-4
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
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

    model = GameTransformer(
        dataset.vocabs, block_size=BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, dropout=DROPOUT
    ).to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            train_loss = estimate_loss(model, train_batcher, EVAL_ITERS, generator)
            val_loss = estimate_loss(model, val_batcher, EVAL_ITERS, generator)
            print(f"step {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        batch = to_device(train_batcher.get_batch(BATCH_SIZE, generator=generator), DEVICE)
        _, loss = model(batch, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n--- sample rollout from the trained model (step 4: does generate() look plausible?) ---")
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
