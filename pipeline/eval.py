"""
Richer validation metrics for GameTransformer's rare-event heads, computed
from an already-trained checkpoint -- no retraining involved.

Why this exists: top-1 accuracy (train.py's estimate_accuracy) is close to
uninformative for touchdown/turnover. Both are so skewed (baseline ~96-98%)
that "always predict no" already matches baseline almost exactly -- exactly
what was observed after a 40000-step run, where accuracy stayed pinned to
baseline. That number alone can't distinguish "model learned nothing" from
"model learned something real but top-1 can't see it." This adds precision/
recall/F1 (positive class = touchdown/turnover occurred) and Brier score
(all four heads), which are far more informative on imbalanced targets.

Sampling caveat: like train.py's own estimate_accuracy/estimate_loss, this
draws EVAL_BATCHES random game-windows from GameBatcher (with replacement,
not an exhaustive sweep of the validation set) -- same approach the project
already uses, just with more batches and richer metrics per batch.
"""

import torch
from torch.nn import functional as F

from dataset import PlayDataset
from get_batch import GameBatcher, build_game_index
from model import OUTPUT_HEADS, GameTransformer
from train import (
    BATCH_SIZE, BLOCK_SIZE, CACHE_PATH, CHECKPOINT_PATH, DATA_DIR, DEVICE,
    HISTORY_SEASONS, SEED, TRAINING_SEASONS, VAL_FRACTION, _head_mask_name, to_device,
)

EVAL_BATCHES = 500  # forward-pass only (cheap) -- enough sampled windows for stable rare-class counts

BINARY_HEADS = ["touchdown", "turnover"]
MULTICLASS_HEADS = ["yards_gained", "return_yards"]


@torch.no_grad()
def collect_predictions(model, batcher, n_batches, generator):
    model.eval()
    preds = {name: [] for name in OUTPUT_HEADS}
    probs = {name: [] for name in OUTPUT_HEADS}
    targets = {name: [] for name in OUTPUT_HEADS}
    masks = {name: [] for name in OUTPUT_HEADS}

    for _ in range(n_batches):
        batch = to_device(batcher.get_batch(BATCH_SIZE, generator=generator), DEVICE)
        logits, _ = model(batch, batch["targets"])
        for name in OUTPUT_HEADS:
            p = F.softmax(logits[name], dim=-1)
            preds[name].append(p.argmax(dim=-1).reshape(-1).cpu())
            probs[name].append(p.reshape(-1, p.shape[-1]).cpu())
            targets[name].append(batch["targets"][name].reshape(-1).cpu())
            masks[name].append(batch["targets"][_head_mask_name(name)].reshape(-1).cpu())

    model.train()
    return {
        name: {
            "preds": torch.cat(preds[name]),
            "probs": torch.cat(probs[name]),
            "targets": torch.cat(targets[name]),
            "mask": torch.cat(masks[name]).bool(),
        }
        for name in OUTPUT_HEADS
    }


def binary_metrics(data, positive_class=1):
    """Precision/recall/F1 for the positive class, plus Brier score, restricted to applicable rows."""
    preds = data["preds"][data["mask"]]
    targets = data["targets"][data["mask"]]
    probs_pos = data["probs"][data["mask"]][:, positive_class]

    tp = ((preds == positive_class) & (targets == positive_class)).sum().item()
    fp = ((preds == positive_class) & (targets != positive_class)).sum().item()
    fn = ((preds != positive_class) & (targets == positive_class)).sum().item()
    support = (targets == positive_class).sum().item()
    n = targets.numel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    brier = ((probs_pos - (targets == positive_class).float()) ** 2).mean().item()

    return {"precision": precision, "recall": recall, "f1": f1, "brier": brier, "support": support, "n": n}


def multiclass_brier(data):
    """Generalized (multiclass) Brier score: mean squared distance between predicted probs and one-hot target."""
    probs = data["probs"][data["mask"]]
    targets = data["targets"][data["mask"]]
    onehot = F.one_hot(targets, num_classes=probs.shape[-1]).float()
    brier = ((probs - onehot) ** 2).sum(dim=-1).mean().item()
    return {"brier": brier, "n": targets.numel()}


def main():
    generator = torch.Generator().manual_seed(SEED)

    print(f"device: {DEVICE}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print(f"loaded checkpoint from step {checkpoint['step']}, val_loss {checkpoint['val_loss']:.4f}")

    print(f"building dataset (cache: {CACHE_PATH})...")
    dataset = PlayDataset(HISTORY_SEASONS, TRAINING_SEASONS, DATA_DIR, cache_path=CACHE_PATH)

    game_ids = sorted(build_game_index(dataset.examples).keys())
    n_val = max(1, round(len(game_ids) * VAL_FRACTION))
    val_game_ids = game_ids[-n_val:]
    val_batcher = GameBatcher(dataset, BLOCK_SIZE, game_ids=val_game_ids)

    model = GameTransformer(checkpoint["vocabs"], **checkpoint["hyperparameters"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"collecting predictions over {EVAL_BATCHES} sampled validation batches...")
    data = collect_predictions(model, val_batcher, EVAL_BATCHES, generator)

    print("\n--- binary heads: precision/recall/F1 (positive class), Brier score ---")
    for name in BINARY_HEADS:
        m = binary_metrics(data[name])
        base_rate = m["support"] / m["n"]
        print(f"{name:10s} precision={m['precision']*100:5.1f}%  recall={m['recall']*100:5.1f}%  "
              f"f1={m['f1']*100:5.1f}%  brier={m['brier']:.4f}  "
              f"positives={m['support']}/{m['n']} ({base_rate*100:.1f}%)")

    print("\n--- multiclass heads: Brier score ---")
    for name in MULTICLASS_HEADS:
        m = multiclass_brier(data[name])
        print(f"{name:15s} brier={m['brier']:.4f}  n={m['n']}")


if __name__ == "__main__":
    main()
