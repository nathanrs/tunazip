import math

import torch
import torch.nn as nn

from tunazip.detops import functions, gemm, rng
from tunazip.detops.elementwise import rope_tables
from tunazip.trace import tap

INIT_STD = 0.02
INIT_DOMAIN = 1
DROPOUT_DOMAIN = 2


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.score_scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.proj = nn.Parameter(torch.empty(d_model, d_model))
        self.q_norm = nn.Parameter(torch.ones(self.d_head))
        self.k_norm = nn.Parameter(torch.ones(self.d_head))

    def forward(self, x, cos, sin, causal_mask, drop):
        batch, seq, d_model = x.shape
        q, k, v = gemm.linear(x, self.qkv).split(d_model, dim=2)
        q = q.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        q = apply_rope(functions.rms_norm(q, self.q_norm), cos, sin)
        k = apply_rope(functions.rms_norm(k, self.k_norm), cos, sin)
        scores = gemm.matmul(q, k.transpose(-1, -2)) * self.score_scale
        probabilities = drop(functions.masked_softmax(scores, causal_mask))
        out = gemm.matmul(probabilities, v)
        return gemm.linear(out.transpose(1, 2).reshape(batch, seq, d_model), self.proj)


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner):
        super().__init__()
        self.attn_pre_norm = nn.Parameter(torch.ones(d_model))
        self.attn = Attention(d_model, n_head)
        self.attn_post_norm = nn.Parameter(torch.ones(d_model))
        self.ffn_pre_norm = nn.Parameter(torch.ones(d_model))
        self.gate = nn.Parameter(torch.empty(d_inner, d_model))
        self.up = nn.Parameter(torch.empty(d_inner, d_model))
        self.down = nn.Parameter(torch.empty(d_model, d_inner))
        self.ffn_post_norm = nn.Parameter(torch.ones(d_model))

    def forward(self, x, cos, sin, causal_mask, drop):
        attended = self.attn(functions.rms_norm(x, self.attn_pre_norm),
                             cos, sin, causal_mask, drop)
        x = x + drop(functions.rms_norm(attended, self.attn_post_norm))
        normed = functions.rms_norm(x, self.ffn_pre_norm)
        hidden = functions.silu(gemm.linear(normed, self.gate)) * gemm.linear(normed, self.up)
        x = x + drop(functions.rms_norm(gemm.linear(hidden, self.down), self.ffn_post_norm))
        return x


class DetTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seed = config.seed
        self.dropout = config.dropout
        self.dropout_calls = 0
        self.drops_this_forward = 0
        self.register_buffer("dropout_keys", torch.zeros(
            1 + 3 * config.n_layer, dtype=torch.int64), persistent=False)
        self.tok_emb = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        self.emb_norm = nn.Parameter(torch.ones(config.d_model))
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.n_head, config.d_inner)
             for _ in range(config.n_layer)])
        self.norm = nn.Parameter(torch.ones(config.d_model))
        self.head = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        cos, sin = rope_tables(config.ctx_len, config.d_model // config.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(config.ctx_len, config.ctx_len, dtype=torch.bool), diagonal=1),
            persistent=False)
        residual_std = INIT_STD / math.sqrt(2 * config.n_layer)
        with torch.no_grad():
            for index, (name, parameter) in enumerate(self.named_parameters()):
                if name.endswith("norm"):
                    continue
                std = residual_std if name.endswith(("proj", "down")) else INIT_STD
                parameter.copy_(rng.normal(
                    rng.stream_key(self.seed, INIT_DOMAIN, index), parameter.shape, std))

    def next_dropout_keys(self):
        keys = [rng.stream_key(self.seed, DROPOUT_DOMAIN, self.dropout_calls + index)
                for index in range(len(self.dropout_keys))]
        self.dropout_calls += len(keys)
        self.dropout_keys.copy_(torch.tensor(keys, dtype=torch.int64))

    def drop(self, x):
        if not self.training or self.dropout == 0.0:
            return x
        key = self.dropout_keys[self.drops_this_forward]
        self.drops_this_forward += 1
        return functions.dropout(x, key, self.dropout)

    def forward(self, idx, output_positions):
        self.drops_this_forward = 0
        seq = idx.size(1)
        cos, sin = self.rope_cos[:seq], self.rope_sin[:seq]
        causal_mask = self.causal_mask[:seq, :seq]
        x = self.drop(functions.rms_norm(
            functions.embedding(self.tok_emb, idx), self.emb_norm))
        tap("emb_norm", x)
        for index, block in enumerate(self.blocks):
            x = block(x, cos, sin, causal_mask, self.drop)
            tap(f"b{index}.ffn_residual", x)
        x = functions.rms_norm(x, self.norm)[:, output_positions]
        tap("final_norm", x)
        return gemm.linear(x, self.head)
