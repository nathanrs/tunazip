import math

import torch

from tunazip.detops import fused

SUM_FRACTION_BITS = 40
INT64_SUM_BITS = 62
SCALED_ROUND_SOURCE = """
template <typename T> T scaled_round(T x, T scale) {
    return nearbyint(__dmul_rn(x, scale));
}
"""
GRID_ROUND_SOURCE = """
template <typename T> T grid_round(T x, T scale, T inverse) {
    return __dmul_rn(nearbyint(__dmul_rn(x, scale)), inverse);
}
"""
SCALE_TWICE_SOURCE = """
template <typename T> T scale_twice(T x, T first, T second) {
    return __dmul_rn(__dmul_rn(x, first), second);
}
"""
PRODUCT_SOURCE = """
template <typename T> T product(T a, T b) {
    return __dmul_rn(a, b);
}
"""
PEAK_SCALES_SOURCE = """
template <typename T> void peak_scales(T peak, T& scale, T& inverse) {
    int exponent;
    frexp(peak, &exponent);
    scale = __longlong_as_double(((long long)(BITS - exponent) + 1023LL) << 52);
    inverse = __longlong_as_double(((long long)(exponent - BITS) + 1023LL) << 52);
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


def product(a, b):
    if a.is_cuda:
        return fused.kernel(PRODUCT_SOURCE)(a, b)
    return a.double() * b


def peak_scales(peak, bits):
    if peak.is_cuda:
        return fused.kernel(PEAK_SCALES_SOURCE.replace("BITS", str(bits)), 2)(peak.double())
    shift = bits - torch.frexp(peak).exponent
    return pow2(shift), pow2(-shift)


def peak_of(x, dim):
    return torch.linalg.vector_norm(x, ord=float("inf"), dim=dim, keepdim=True)


def quantize(x, dim, bits):
    scale, inverse = peak_scales(peak_of(x, dim), bits)
    return scaled_round(x, scale), inverse


def grid_round(x, dim, bits):
    scale, inverse = peak_scales(peak_of(x, dim), bits)
    if x.is_cuda:
        return fused.kernel(GRID_ROUND_SOURCE)(x, scale, inverse)
    return torch.round(x.double() * scale) * inverse


def exact_sum(x, dim):
    dims = (dim,) if isinstance(dim, int) else dim
    terms = math.prod(x.shape[d] for d in dims)
    bits = min(SUM_FRACTION_BITS, INT64_SUM_BITS - (terms - 1).bit_length())
    quantized, inverse = quantize(x, dim, bits)
    total = quantized.to(torch.int64).sum(dim=dim, keepdim=True)
    return product(total, inverse)
