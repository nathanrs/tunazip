import torch

UNIFORM_BITS = 24
NORMAL_UNIFORMS = 12


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


def stream_key(*parts):
    key = torch.zeros((), dtype=torch.int64)
    for part in parts:
        key = splitmix64(key ^ torch.tensor(_signed(part % (1 << 64)), dtype=torch.int64))
    return int(key)


def uniform_integers(key, count, device):
    counters = torch.arange(count, dtype=torch.int64, device=device) + key
    return _logical_shift_right(splitmix64(counters), 64 - UNIFORM_BITS)


def keep_mask(key, shape, drop_probability, device):
    count = 1
    for size in shape:
        count *= size
    threshold = round(drop_probability * (1 << UNIFORM_BITS))
    return (uniform_integers(key, count, device) >= threshold).reshape(shape)


def normal(key, shape, std):
    count = 1
    for size in shape:
        count *= size
    total = torch.zeros(count, dtype=torch.int64)
    for index in range(NORMAL_UNIFORMS):
        total += uniform_integers(stream_key(key, index), count, torch.device("cpu"))
    centered = total - (NORMAL_UNIFORMS // 2 << UNIFORM_BITS)
    return (centered.double() * (std / (1 << UNIFORM_BITS))).float().reshape(shape)
