import torch

from model import MaskedMultiHeadSelfAttention, PlayerHistoryEncoder


def test_all_padding_produces_no_nan_and_a_fixed_output():
    torch.manual_seed(0)
    enc = PlayerHistoryEncoder(feature_dim=9, max_history=16, n_embd=32, n_head=2, dropout=0.0)

    history = torch.zeros(3, 16, 9)
    mask = torch.zeros(3, 16, dtype=torch.bool)
    out = enc(history, mask)

    assert not torch.isnan(out).any()
    # a player with zero history should get the SAME learned "no history"
    # default regardless of batch position -- not garbage, not raw zeros.
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[0], out[2])


def test_all_padding_backward_pass_has_no_nan_gradients():
    torch.manual_seed(0)
    enc = PlayerHistoryEncoder(feature_dim=9, max_history=16, n_embd=32, n_head=2, dropout=0.0)
    history = torch.zeros(2, 16, 9)
    mask = torch.zeros(2, 16, dtype=torch.bool)

    enc(history, mask).sum().backward()

    for p in enc.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_mixed_padding_ignores_padding_positions():
    """
    Two players with identical real history but different amounts of trailing
    padding should produce the same encoded output -- padding positions must
    not leak into the pooled result.
    """
    torch.manual_seed(0)
    enc = PlayerHistoryEncoder(feature_dim=9, max_history=16, n_embd=32, n_head=2, dropout=0.0)
    enc.eval()

    real_weeks = torch.randn(5, 9)
    history_a = torch.zeros(16, 9)
    history_a[:5] = real_weeks
    mask_a = torch.zeros(16, dtype=torch.bool)
    mask_a[:5] = True

    history_b = history_a.clone()
    history_b[5:] = torch.randn(11, 9) * 100  # garbage in the padding region
    mask_b = mask_a.clone()  # same mask -- garbage positions still marked as padding

    out_a = enc(history_a.unsqueeze(0), mask_a.unsqueeze(0))
    out_b = enc(history_b.unsqueeze(0), mask_b.unsqueeze(0))

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_masked_attention_key_padding_mask_shape():
    torch.manual_seed(0)
    attn = MaskedMultiHeadSelfAttention(n_embd=16, num_heads=2, dropout=0.0)
    x = torch.randn(4, 8, 16)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, 5:] = False

    out = attn(x, mask)
    assert out.shape == (4, 8, 16)
    assert not torch.isnan(out).any()
