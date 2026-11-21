from functools import cache

import torch

from detzip.detops import fused

UNIFORM_BITS = 24
NORMAL_UNIFORMS = 12
UINT64_MASK = (1 << 64) - 1
KEEP_SOURCE = """
template <typename T> T keep(T counter, T key) {
    unsigned long long z = (unsigned long long)counter + (unsigned long long)key + GOLDENULL;
    z = (z ^ (z >> 30)) * MIX_1ULL;
    z = (z ^ (z >> 27)) * MIX_2ULL;
    z = z ^ (z >> 31);
    return (T)((z >> UNIFORM_SHIFT) >= THRESHOLDULL);
}
"""
APPLY_KEEP_SOURCE = """
template <typename T> T apply_keep(T x, T keep) {
    return x * (keep != 0 ? (T)SCALE : (T)0.0);
}
"""


def _signed(value):
    return value - (1 << 64) if value >= (1 << 63) else value


GOLDEN = _signed(0x9E3779B97F4A7C15)
MIX_1 = _signed(0xBF58476D1CE4E5B9)
MIX_2 = _signed(0x94D049BB133111EB)


def _logical_shift_right(x, bits):
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def splitmix64(x):
    z = x + GOLDEN
    z = (z ^ _logical_shift_right(z, 30)) * MIX_1
    z = (z ^ _logical_shift_right(z, 27)) * MIX_2
    return z ^ _logical_shift_right(z, 31)


def host_splitmix64(x):
    z = (x + GOLDEN) & UINT64_MASK
    z = ((z ^ (z >> 30)) * MIX_1) & UINT64_MASK
    z = ((z ^ (z >> 27)) * MIX_2) & UINT64_MASK
    return z ^ (z >> 31)


def stream_key(*parts):
    key = 0
    for part in parts:
        key = host_splitmix64(key ^ (part & UINT64_MASK))
    return _signed(key)


def uniform_integers(key, count, device):
    counters = torch.arange(count, dtype=torch.int64, device=device) + key
    return _logical_shift_right(splitmix64(counters), 64 - UNIFORM_BITS)


@cache
def counters(count, device):
    return torch.arange(count, dtype=torch.int64, device=device)


def dropout(x, key, drop_probability):
    threshold = round(drop_probability * (1 << UNIFORM_BITS))
    if x.is_cuda:
        keep = fused.kernel(
            KEEP_SOURCE.replace("GOLDEN", hex(GOLDEN & UINT64_MASK))
            .replace("MIX_1", hex(MIX_1 & UINT64_MASK)).replace("MIX_2", hex(MIX_2 & UINT64_MASK))
            .replace("UNIFORM_SHIFT", str(64 - UNIFORM_BITS))
            .replace("THRESHOLD", str(threshold)))(counters(x.numel(), x.device), key)
        scale = torch.tensor(1.0 / (1.0 - drop_probability), dtype=torch.float32).item()
        return fused.kernel(APPLY_KEEP_SOURCE.replace("SCALE", fused.literal(scale)))(
            x, keep.reshape(x.shape))
    keep = (uniform_integers(key, x.numel(), x.device) >= threshold).reshape(x.shape)
    return x * (keep.float() * (1.0 / (1.0 - drop_probability)))


def normal(key, shape, std):
    count = 1
    for size in shape:
        count *= size
    total = torch.zeros(count, dtype=torch.int64)
    for index in range(NORMAL_UNIFORMS):
        total += uniform_integers(stream_key(key, index), count, torch.device("cpu"))
    centered = total - (NORMAL_UNIFORMS // 2 << UNIFORM_BITS)
    return (centered.double() * (std / (1 << UNIFORM_BITS))).float().reshape(shape)
