import torch
import pytest

from dataset import PlayDataset
from get_batch import GameBatcher
from model import GameTransformer


@pytest.fixture(scope="module")
def small_dataset():
    return PlayDataset(history_seasons=[2022], training_seasons=[2023], data_dir="../data", max_examples=1500)


def _tiny_model(vocabs, block_size):
    return GameTransformer(vocabs, block_size=block_size, n_embd=32, n_head=2, n_layer=2, dropout=0.0)


def test_forward_pass_shapes_and_no_nan(small_dataset):
    torch.manual_seed(0)
    batch_size, block_size = 4, 8
    batcher = GameBatcher(small_dataset, block_size=block_size)
    batch = batcher.get_batch(batch_size, generator=torch.Generator().manual_seed(0))
    model = _tiny_model(small_dataset.vocabs, block_size)

    logits, loss = model(batch, batch["targets"])

    assert logits["yards_gained"].shape == (batch_size, block_size, len(small_dataset.vocabs["yards_gained"]))
    assert logits["touchdown"].shape == (batch_size, block_size, 2)
    assert logits["turnover"].shape == (batch_size, block_size, 2)
    assert logits["return_yards"].shape == (batch_size, block_size, len(small_dataset.vocabs["return_yards"]))
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_forward_pass_without_targets_returns_no_loss(small_dataset):
    batch_size, block_size = 2, 8
    batcher = GameBatcher(small_dataset, block_size=block_size)
    batch = batcher.get_batch(batch_size, generator=torch.Generator().manual_seed(1))
    model = _tiny_model(small_dataset.vocabs, block_size)

    logits, loss = model(batch)

    assert loss is None
    assert logits["yards_gained"].shape[:2] == (batch_size, block_size)


def test_model_overfits_a_tiny_fixed_batch(small_dataset):
    """
    The brief's flagged risk is whether nested attention inside a causal
    sequence transformer trains sensibly at all. This doesn't prove the full
    architecture generalizes -- it just confirms the forward/backward/masked
    -loss plumbing is correct by checking loss goes down, and never NaNs,
    when repeatedly optimizing the exact same small batch.
    """
    torch.manual_seed(0)
    batch_size, block_size = 4, 8
    batcher = GameBatcher(small_dataset, block_size=block_size)
    batch = batcher.get_batch(batch_size, generator=torch.Generator().manual_seed(0))

    model = _tiny_model(small_dataset.vocabs, block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(60):
        _, loss = model(batch, batch["targets"])
        assert not torch.isnan(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"loss didn't drop: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_loss_weights_default_matches_uniform_ones(small_dataset):
    torch.manual_seed(0)
    batch_size, block_size = 4, 8
    batcher = GameBatcher(small_dataset, block_size=block_size)
    batch = batcher.get_batch(batch_size, generator=torch.Generator().manual_seed(0))

    model_none = _tiny_model(small_dataset.vocabs, block_size)  # loss_weights defaults to None
    model_ones = GameTransformer(
        small_dataset.vocabs, block_size=block_size, n_embd=32, n_head=2, n_layer=2, dropout=0.0,
        loss_weights={"yards_gained": 1.0, "touchdown": 1.0, "turnover": 1.0, "return_yards": 1.0},
    )
    model_ones.load_state_dict(model_none.state_dict())

    _, loss_none = model_none(batch, batch["targets"])
    _, loss_ones = model_ones(batch, batch["targets"])

    assert loss_none.item() == pytest.approx(loss_ones.item())


def test_loss_weights_change_total_loss(small_dataset):
    torch.manual_seed(0)
    batch_size, block_size = 4, 8
    batcher = GameBatcher(small_dataset, block_size=block_size)
    batch = batcher.get_batch(batch_size, generator=torch.Generator().manual_seed(0))

    model_default = _tiny_model(small_dataset.vocabs, block_size)
    model_weighted = GameTransformer(
        small_dataset.vocabs, block_size=block_size, n_embd=32, n_head=2, n_layer=2, dropout=0.0,
        loss_weights={"yards_gained": 1.0, "touchdown": 5.0, "turnover": 1.0, "return_yards": 1.0},
    )
    model_weighted.load_state_dict(model_default.state_dict())

    _, loss_default = model_default(batch, batch["targets"])
    _, loss_weighted = model_weighted(batch, batch["targets"])

    assert loss_weighted.item() != pytest.approx(loss_default.item())
