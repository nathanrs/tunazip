import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detzip import trace
from detzip.detmodel import DetTransformer
from detzip.detops import fixed as det_fixed
from detzip.detops import functions as det_functions
from detzip.detops import optim as det_optim
from detzip.detops.elementwise import host_cos_of_pi_times
from detzip.arithmetic_coder import (ArithmeticDecoder, ArithmeticEncoder,
                                     BitInputStream, BitOutputStream)
from detzip.trace import tap

LOG2 = math.log(2.0)
FREQ_TOTAL = 10_000_000


@dataclass
class Config:
    vocab_size: int
    n_layer: int = 12
    d_model: int = 1024
    n_head: int = 8
    d_inner: int = 2816
    ctx_len: int = 192
    dropout: float = 0.25
    batch_size: int = 64
    block_len: int = 500_000
    lr: float = 4e-5
    min_lr_ratio: float = 0.25
    warmup: int = 200
    clip: float = 0.25
    beta1: float = 0.0
    beta2: float = 0.9999
    adam_eps: float = 1e-9
    retrain_period: int = 1
    retrain_len: int = 35_000_000
    retrain_batch_size: int = 32
    retrain_update_len: int = 64
    retrain_lr: str = "1.6e-4,13000,1.6e-4,93000,7.9e-5,163000,4.0e-5,1911300,1.3e-5"
    dtype: str = "bf16"
    seed: int = 1111
    det: bool = False
    matmul_bits: int = 16


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return normed.type_as(x) * self.weight


def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

    def forward(self, x, cos, sin, name):
        batch, seq, d_model = x.shape
        qkv = self.qkv(x)
        tap(f"{name}.qkv", qkv)
        q, k, v = qkv.split(d_model, dim=2)
        q = q.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        tap(f"{name}.q_norm", q)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        tap(f"{name}.q_rope", q)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        tap(f"{name}.sdpa", out)
        out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        out = self.proj(out)
        tap(f"{name}.proj", out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_inner):
        super().__init__()
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_model, d_inner, bias=False)
        self.down = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x, name):
        gate = self.gate(x)
        tap(f"{name}.gate", gate)
        hidden = F.silu(gate) * self.up(x)
        tap(f"{name}.silu_mul_up", hidden)
        out = self.down(hidden)
        tap(f"{name}.down", out)
        return out


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout):
        super().__init__()
        self.attn_pre_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_head, dropout)
        self.attn_post_norm = RMSNorm(d_model)
        self.ffn_pre_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_inner)
        self.ffn_post_norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, name):
        normed = self.attn_pre_norm(x)
        tap(f"{name}.attn_pre_norm", normed)
        x = x + self.drop(self.attn_post_norm(
            self.attn(normed, cos, sin, f"{name}.attn")))
        tap(f"{name}.attn_residual", x)
        x = x + self.drop(self.ffn_post_norm(
            self.ffn(self.ffn_pre_norm(x), f"{name}.ffn")))
        tap(f"{name}.ffn_residual", x)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_norm = RMSNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.n_head, config.d_inner, config.dropout)
             for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        cos, sin = build_rope_cache(config.ctx_len, config.d_model // config.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, output_positions):
        seq = idx.size(1)
        cos, sin = self.rope_cos[:seq], self.rope_sin[:seq]
        x = self.drop(self.emb_norm(self.tok_emb(idx)))
        tap("emb_norm", x)
        for index, block in enumerate(self.blocks):
            x = block(x, cos, sin, f"b{index}")
        x = self.norm(x)[:, output_positions]
        tap("final_norm", x)
        return self.head(x)


def lr_at(step, peak_lr, min_lr, warmup, total_steps, cos_of_pi_times):
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + cos_of_pi_times(progress))


def parse_piecewise_lr(value):
    parts = str(value).split(",")
    schedule = [(0, float(parts[0]))]
    for index in range(1, len(parts), 2):
        schedule.append((int(parts[index]), float(parts[index + 1])))
    return schedule


def piecewise_lr_at(step, schedule):
    if step <= schedule[0][0]:
        return schedule[0][1]
    for (start_step, start_lr), (end_step, end_lr) in zip(schedule, schedule[1:]):
        if step <= end_step:
            fraction = (step - start_step) / (end_step - start_step)
            return start_lr + fraction * (end_lr - start_lr)
    return schedule[-1][1]


def float_cos_of_pi_times(fraction):
    return math.cos(math.pi * fraction)


def make_freq_table(logits, det):
    if det:
        prob = det_functions.softmax64(logits.detach())
    else:
        prob = F.softmax(logits.detach(), dim=-1)
    freq = torch.clamp(torch.round(prob * FREQ_TOTAL).to(torch.int32), min=1)
    return torch.cumsum(freq, dim=-1)


def set_lr(optimizer, lr):
    if isinstance(optimizer, det_optim.Adam):
        optimizer.lr = lr
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


class Codec:
    def __init__(self, config, device, tracer=None, digest_every=0, log=print):
        self.config = config
        self.device = device
        self.tracer = tracer
        self.digest_every = digest_every
        self.digests = []
        self.log = log
        torch.use_deterministic_algorithms(not config.det)
        det_fixed.MATMUL_BITS = config.matmul_bits
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        model_class = DetTransformer if config.det else Transformer
        adam = det_optim.Adam if config.det else torch.optim.Adam
        self.model = model_class(config).to(device)
        self.model.eval()
        betas = (config.beta1, config.beta2)
        self.retrain_schedule = parse_piecewise_lr(config.retrain_lr)
        self.optimizer = adam(
            self.model.parameters(), lr=config.lr, betas=betas, eps=config.adam_eps)
        self.retrain_optimizer = adam(
            self.model.parameters(), lr=self.retrain_schedule[0][1], betas=betas,
            eps=config.adam_eps)
        self.amp = device.type == "cuda" and config.dtype == "bf16" and not config.det
        self.cos_of_pi_times = host_cos_of_pi_times if config.det else float_cos_of_pi_times
        self.retrain_step = 0

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.amp)

    def train_on(self, logits, targets, optimizer, traced):
        optimizer.zero_grad(set_to_none=True)
        if self.config.det:
            loss = det_functions.cross_entropy(logits, targets)
            loss.backward()
            grad_norm, scale = det_optim.gradient_scale(
                list(self.model.parameters()), self.config.clip)
        else:
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip)
        if traced:
            tap("loss", loss)
            for name, parameter in self.model.named_parameters():
                tap(f"grad.{name}", parameter.grad)
            tap("grad_norm", grad_norm)
        if self.config.det:
            optimizer.step(scale)
        else:
            optimizer.step()
        if traced:
            for name, parameter in self.model.named_parameters():
                tap(f"param.{name}", parameter)
        return loss

    def retrain(self, file_data, file_pos):
        config = self.config
        lo = max(0, file_pos - config.retrain_len)
        retrain_batch = config.retrain_batch_size or config.batch_size
        stride = (file_pos - lo) // retrain_batch
        if stride < 2:
            return
        stream_base = lo + np.arange(retrain_batch) * stride
        n_targets = stride - 1
        update_len = config.retrain_update_len or config.ctx_len
        self.model.train()
        nats = 0.0
        for start in range(0, n_targets, update_len):
            length = min(update_len, n_targets - start)
            context_start = (max(0, start - (config.ctx_len - length))
                             if config.retrain_update_len else start)
            prefix_len = start - context_start
            cols = context_start + np.arange(prefix_len + length)
            inp = torch.from_numpy(
                file_data[stream_base[:, None] + cols[None, :]]).to(self.device)
            target_cols = start + np.arange(length)
            targets = torch.from_numpy(
                file_data[stream_base[:, None] + target_cols[None, :] + 1]).to(self.device)
            if config.det:
                self.model.next_dropout_keys()
            with self.autocast():
                logits = self.model(inp, slice(prefix_len, prefix_len + length))
            set_lr(self.retrain_optimizer,
                   piecewise_lr_at(self.retrain_step, self.retrain_schedule))
            loss = self.train_on(
                logits.float().reshape(-1, config.vocab_size), targets.reshape(-1),
                self.retrain_optimizer, traced=False)
            self.retrain_step += 1
            nats += loss.item() * length
        self.model.eval()
        self.log(f"replay at {file_pos}: window {file_pos - lo}, "
                 f"{nats / n_targets / LOG2:.3f} bits/token, step {self.retrain_step}")

    def run(self, file_data, original_len, encoder=None, decoder=None):
        config = self.config
        batch = config.batch_size
        file_len = ((original_len + batch - 1) // batch) * batch
        if len(file_data) < file_len:
            file_data = np.concatenate(
                [file_data, np.zeros(file_len - len(file_data), dtype=np.int64)])
        block_len = max(batch, (config.block_len // batch) * batch)
        total_steps = file_len // batch
        min_lr = config.lr * config.min_lr_ratio
        n_bits = 0.0
        train_step = 0
        block_start = 0
        last_retrain_pos = 0
        started = time.time()
        while block_start < file_len:
            if config.retrain_period and block_start - last_retrain_pos >= config.retrain_period:
                self.retrain(file_data, block_start)
                last_retrain_pos = block_start
            cur_block_len = min(file_len - block_start, block_len)
            block_stride = cur_block_len // batch
            stream_base = block_start + np.arange(batch) * block_stride
            for stream_pos in range(block_stride):
                traced = self.tracer is not None and train_step < self.tracer.trace_steps
                if traced:
                    self.tracer.begin(train_step)
                window = min(stream_pos, config.ctx_len)
                if window == 0:
                    inp = np.zeros((batch, 1), dtype=np.int64)
                else:
                    offsets = stream_pos - window + np.arange(window)
                    inp = file_data[stream_base[:, None] + offsets[None, :]]
                set_lr(self.optimizer,
                       lr_at(train_step, config.lr, min_lr, config.warmup, total_steps,
                             self.cos_of_pi_times))
                with self.autocast():
                    logits = self.model(torch.from_numpy(inp).to(self.device),
                                        slice(-1, None))
                code_logits = logits[:, -1, :].float()
                tap("logits", code_logits)
                code_pos = stream_base + stream_pos
                freq = None
                if encoder is not None or decoder is not None or traced or config.det:
                    freq = make_freq_table(code_logits, config.det)
                    tap("freq_table", freq)
                    freq = freq.cpu().numpy()
                    for i in range(batch):
                        if decoder is not None:
                            file_data[code_pos[i]] = decoder.read(freq[i])
                        elif encoder is not None:
                            encoder.write(freq[i], int(file_data[code_pos[i]]))
                targets = torch.from_numpy(file_data[code_pos]).to(self.device)
                if config.det:
                    counts = np.diff(freq, prepend=0, axis=1)[np.arange(batch), file_data[code_pos]]
                    n_bits += float(np.log2(freq[:, -1] / counts).sum())
                else:
                    n_bits += F.cross_entropy(
                        code_logits, targets, reduction="sum").item() / LOG2
                self.train_on(code_logits, targets, self.optimizer, traced)
                if self.digest_every and train_step % self.digest_every == 0:
                    self.digests.append({
                        "step": train_step, "logits": trace.sha(code_logits),
                        "freq": None if freq is None else trace.sha(torch.from_numpy(freq)),
                        "params": trace.parameter_digest(self.model)})
                if traced:
                    self.tracer.end()
                train_step += 1
            block_start += cur_block_len
            elapsed = time.time() - started
            self.log(f"block end {block_start}/{file_len}: {n_bits / 8:.0f} bytes, "
                     f"{n_bits / block_start:.3f} bits/token, "
                     f"{block_start / elapsed:.0f} tokens/s")
        return {"tokens": original_len, "coded_symbols": file_len,
                "modeled_bits": n_bits, "modeled_bytes": n_bits / 8,
                "steps": train_step, "retrain_steps": self.retrain_step,
                "seconds": time.time() - started,
                "final_params": trace.parameter_digest(self.model)}, file_data


def measure(config, tokens, device, **codec_options):
    codec = Codec(config, device, **codec_options)
    result, _ = codec.run(tokens.copy(), len(tokens))
    return result, codec


def encode(config, tokens, payload_path, device, **codec_options):
    codec = Codec(config, device, **codec_options)
    header = json.dumps({"config": asdict(config), "tokens": len(tokens)}).encode()
    with open(payload_path, "wb") as file:
        file.write(len(header).to_bytes(4, "big") + header)
        bit_out = BitOutputStream(file)
        encoder = ArithmeticEncoder(32, bit_out)
        result, _ = codec.run(tokens.copy(), len(tokens), encoder=encoder)
        encoder.finish()
        bit_out.close()
    return result, codec


def decode(payload_path, device, **codec_options):
    with open(payload_path, "rb") as file:
        header = json.loads(file.read(int.from_bytes(file.read(4), "big")))
        config = Config(**header["config"])
        codec = Codec(config, device, **codec_options)
        decoder = ArithmeticDecoder(32, BitInputStream(file))
        result, file_data = codec.run(
            np.zeros(header["tokens"], dtype=np.int64), header["tokens"], decoder=decoder)
    return file_data[:header["tokens"]], result, codec


def load_tokens(path):
    with open(path, "rb") as file:
        return np.frombuffer(file.read(), dtype=">u2").astype(np.int64)
