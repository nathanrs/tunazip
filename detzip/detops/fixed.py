import torch

from detzip.detops import fused

MATMUL_BITS = 16
SUM_FRACTION_BITS = 40
SUM_MAX_TERMS = 2 ** 22
SCALED_ROUND_SOURCE = """
template <typename T> T scaled_round(T x, T scale) {
    return nearbyint(__dmul_rn(x, scale));
}
"""
SCALE_TWICE_SOURCE = """
template <typename T> T scale_twice(T x, T first, T second) {
    return __dmul_rn(__dmul_rn(x, first), second);
}
"""


def pow2(exponent):
    return ((exponent.to(torch.int64) + 1023) << 52).view(torch.float64)


def scaled_round(x, scale):
    if x.is_cuda:
        return fused.kernel(SCALED_ROUND_SOURCE)(x, scale)
    return torch.round(x.double() * scale)


def scale_twice(x, first, second):
    if x.is_cuda:
        return fused.kernel(SCALE_TWICE_SOURCE)(x, first, second)
    return x * first * second


def quantize(x, dim, bits):
    peak = x.abs().amax(dim=dim, keepdim=True)
    shift = bits - torch.frexp(peak).exponent
    return scaled_round(x, pow2(shift)), shift


def exact_sum(x, dim):
    quantized, shift = quantize(x, dim, SUM_FRACTION_BITS)
    if quantized.numel() // shift.numel() > SUM_MAX_TERMS:
        raise ValueError("too many terms for an exact int64 sum")
    total = quantized.to(torch.int64).sum(dim=dim, keepdim=True)
    return total.double() * pow2(-shift)
