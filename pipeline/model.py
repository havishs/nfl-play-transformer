"""
Model v0: nested play-level attention (offense-11 x defense-11 bidirectional
cross-attention -> one play-summary token per play) feeding an outer causal
decoder (adapted from the Karpathy "Let's build GPT" backbone, originally
~/Downloads/gpt.py) over play-summary tokens, with masked-loss output heads
for yards_gained/touchdown/turnover/return_yards.

Per PROJECT_BRIEF.md's staging: player history (design point 1) now has a
real learned encoder (PlayerHistoryEncoder, below) -- see the design doc's
addendum for why it's wired in ADDITIVELY (summed into PlayerEncoder's
per-player embedding) rather than as true encoder-decoder cross-attention
as the brief literally describes: same functional goal, much less risk
right after getting the base architecture to finally train reliably.
In-game running form (design point 2) now has a real causal EMA
(TeamFormEncoder, below, backed by team_form.py) wired into the forward
pass, fed by both build_dataset.py (real outcomes) and generate.py's
rollout (the model's own sampled outcomes) via the same update function.

play_type is intentionally NOT an output head: it's currently only a
situational *input* field in build_dataset.py's build_targets() (never a
target), and adding it as an output without restructuring the input/target
split would leak -- the decoder token already has play_type baked into its
input embedding via SituationalEncoder, so "predicting" it back out would
just be inverting that lookup. Deferred to a later pass.

No generate() yet either -- autoregressive rollout needs game-state
transition logic (down progression, score/field-position updates from a
generated outcome) that's a separate, bigger piece of work than v0's scope
of "does the forward pass + masked loss train sensibly at all."
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset import (
    HISTORY_FEATURE_DIM,
    MAX_HISTORY,
    PERSONNEL_FEATURE_DIM,
    SITUATIONAL_FIELDS,
    TEAM_FIELDS,
    TEAM_FORM_FEATURE_DIM,
)

OUTPUT_HEADS = ["yards_gained", "touchdown", "turnover", "return_yards"]


class Head(nn.Module):
    """ One head of causal self-attention, over the outer (play-level) sequence. """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ Multiple causal self-attention heads in parallel, over the play-level sequence. """

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Causal self-attention over the play-level sequence, then feedforward. """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CrossHead(nn.Module):
    """ One head of NON-causal cross-attention -- within-play matchup attention, not sequential. """

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in, kv_in):
        # q_in: (..., Tq, C), kv_in: (..., Tkv, C); no causal mask
        q = self.query(q_in)
        k = self.key(kv_in)
        v = self.value(kv_in)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([CrossHead(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in, kv_in):
        out = torch.cat([h(q_in, kv_in) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class MaskedSelfHead(nn.Module):
    """ One head of non-causal self-attention with a key-padding mask (for the history encoder). """

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        # x: (..., T, C); key_padding_mask: (..., T) bool, True = real/attendable key.
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (..., T, T)
        wei = wei.masked_fill(~key_padding_mask.unsqueeze(-2), float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # a query position with NO real keys (a player with zero history) softmaxes
        # an all -inf row to NaN -- zero it instead of letting NaN propagate. Harmless:
        # that query's output is excluded from the encoder's final masked mean-pool anyway.
        wei = torch.nan_to_num(wei, nan=0.0)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([MaskedSelfHead(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        out = torch.cat([h(x, key_padding_mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class PlayerHistoryEncoder(nn.Module):
    """
    Bidirectional (non-causal) self-attention over a player's raw per-week
    history -- a real learned encoder replacing PlayerFeatureLookup.lookup()'s
    hand-computed causal mean, per PROJECT_BRIEF.md's design point 1.
    Non-causal is correct here even though the outer decoder is causal: a
    player's PAST history is already fully "finished" by the time it's being
    summarized for the current play, unlike the play sequence itself.

    Shared (same weights) across offense and defense calls -- the underlying
    9-dim [off_feats | def_feats] vector is already side-agnostic (whichever
    half doesn't apply is zero-filled), matching PlayerEncoder's own reuse.
    """

    def __init__(self, feature_dim, max_history, n_embd, n_head, dropout):
        super().__init__()
        self.week_proj = nn.Linear(feature_dim, n_embd)
        self.position_embedding = nn.Embedding(max_history, n_embd)
        self.attn = MaskedMultiHeadSelfAttention(n_embd, n_head, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, history, mask):
        # history: (..., T, feature_dim), mask: (..., T) bool -> (..., n_embd)
        T = history.shape[-2]
        pos = self.position_embedding(torch.arange(T, device=history.device))
        x = self.week_proj(history) + pos
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))

        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=-2) / mask_f.sum(dim=-2).clamp(min=1)
        return self.out_proj(pooled)


class PlayerEncoder(nn.Module):
    """ Position embedding + linear projection of hand-computed causal features, additively combined. """

    def __init__(self, num_positions, feature_dim, n_embd):
        super().__init__()
        self.position_embedding = nn.Embedding(num_positions, n_embd)
        self.feature_proj = nn.Linear(feature_dim, n_embd)

    def forward(self, position_idx, features):
        # position_idx: (..., 11) long, features: (..., 11, feature_dim) -> (..., 11, n_embd)
        return self.position_embedding(position_idx) + self.feature_proj(features)


class NestedPlayAttention(nn.Module):
    """
    offense-11 x defense-11 bidirectional cross-attention (no same-side
    self-attention -- the point is modeling matchups, not intra-team
    chemistry) -> mean-pool each side -> concat -> project to one
    play-summary token.
    """

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln_off = nn.LayerNorm(n_embd)
        self.ln_def = nn.LayerNorm(n_embd)
        self.off_attends_def = MultiHeadCrossAttention(n_embd, n_head, dropout)
        self.def_attends_off = MultiHeadCrossAttention(n_embd, n_head, dropout)
        self.summary_proj = nn.Linear(2 * n_embd, n_embd)

    def forward(self, offense, defense):
        # offense/defense: (..., 11, n_embd) -> (..., n_embd)
        off_n = self.ln_off(offense)
        def_n = self.ln_def(defense)
        off_ctx = offense + self.off_attends_def(off_n, def_n)
        def_ctx = defense + self.def_attends_off(def_n, off_n)
        off_pool = off_ctx.mean(dim=-2)
        def_pool = def_ctx.mean(dim=-2)
        return self.summary_proj(torch.cat([off_pool, def_pool], dim=-1))


class SituationalEncoder(nn.Module):
    """ Sums one embedding per situational/team field into a single n_embd vector. """

    FIELDS = SITUATIONAL_FIELDS + TEAM_FIELDS  # must match dataset.py's _situational_tensor field order

    def __init__(self, vocabs, n_embd):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(vocabs["team"] if f in TEAM_FIELDS else vocabs[f]), n_embd)
            for f in self.FIELDS
        ])

    def forward(self, situational_idx):
        # situational_idx: (..., len(FIELDS)) long -> (..., n_embd)
        return sum(self.embeddings[i](situational_idx[..., i]) for i in range(len(self.FIELDS)))


class TeamFormEncoder(nn.Module):
    """
    Linear projection of the causal in-game team-form feature vector (see
    team_form.py). Note the input is NOT normalized -- yards_ema can range
    to +-75 on an explosive play while touchdown_ema/turnover_ema and the
    two history flags stay in [0, 1], a real scale imbalance relative to
    e.g. PlayerEncoder's roughly [-3, 3] EPA-based features. A single
    Linear can in principle learn to downweight the large-magnitude input,
    but this may slow how quickly this head finds sane weights -- worth
    watching in early training loss curves, not yet worth a normalization
    pass on its own.
    """

    def __init__(self, feature_dim, n_embd):
        super().__init__()
        self.proj = nn.Linear(feature_dim, n_embd)

    def forward(self, team_form):
        # team_form: (..., feature_dim) -> (..., n_embd)
        return self.proj(team_form)


class GameTransformer(nn.Module):
    def __init__(self, vocabs, block_size, n_embd=128, n_head=4, n_layer=4, dropout=0.1, loss_weights=None):
        super().__init__()
        self.block_size = block_size
        self.loss_weights = loss_weights if loss_weights is not None else {name: 1.0 for name in OUTPUT_HEADS}
        self.player_encoder = PlayerEncoder(len(vocabs["position"]), PERSONNEL_FEATURE_DIM, n_embd)
        self.history_encoder = PlayerHistoryEncoder(HISTORY_FEATURE_DIM, MAX_HISTORY, n_embd, n_head, dropout)
        self.nested_attention = NestedPlayAttention(n_embd, n_head, dropout)
        self.situational_encoder = SituationalEncoder(vocabs, n_embd)
        self.team_form_encoder = TeamFormEncoder(TEAM_FORM_FEATURE_DIM, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        self.heads = nn.ModuleDict({
            "yards_gained": nn.Linear(n_embd, len(vocabs["yards_gained"])),
            "touchdown": nn.Linear(n_embd, 2),
            "turnover": nn.Linear(n_embd, 2),
            "return_yards": nn.Linear(n_embd, len(vocabs["return_yards"])),
        })

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch, targets=None):
        offense = self.player_encoder(batch["offense_position"], batch["offense_features"]) \
            + self.history_encoder(batch["offense_history"], batch["offense_history_mask"])
        defense = self.player_encoder(batch["defense_position"], batch["defense_features"]) \
            + self.history_encoder(batch["defense_history"], batch["defense_history_mask"])
        play_summary = self.nested_attention(offense, defense)  # (B, T, n_embd)

        situational = self.situational_encoder(batch["situational"])  # (B, T, n_embd)
        team_form = self.team_form_encoder(batch["team_form"])  # (B, T, n_embd)
        T = batch["situational"].shape[1]
        pos = self.position_embedding(torch.arange(T, device=batch["situational"].device))  # (T, n_embd)

        x = play_summary + situational + team_form + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = {name: head(x) for name, head in self.heads.items()}

        if targets is None:
            return logits, None

        total_loss = torch.zeros((), device=x.device)
        for name in OUTPUT_HEADS:
            head_logits = logits[name]
            B, T, C = head_logits.shape
            per_position_loss = F.cross_entropy(
                head_logits.view(B * T, C), targets[name].view(B * T), reduction="none"
            )
            mask_name = "td_turnover_mask" if name in ("touchdown", "turnover") else f"{name}_mask"
            mask = targets[mask_name].view(B * T)
            head_loss = (per_position_loss * mask).sum() / mask.sum().clamp(min=1)
            total_loss = total_loss + self.loss_weights[name] * head_loss

        return logits, total_loss
