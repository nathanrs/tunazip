import torch

MATMUL_BITS = 16
SUM_FRACTION_BITS = 40
SUM_MAX_TERMS = 2 ** 22


def pow2(exponent):
    return ((exponent.to(torch.int64) + 1023) << 52).view(torch.float64)


def quantize(x, dim, bits):
    peak = x.abs().amax(dim=dim, keepdim=True)
    shift = bits - torch.frexp(peak).exponent
    return torch.round(x.double() * pow2(shift)), shift


def exact_sum(x, dim):
    quantized, shift = quantize(x, dim, SUM_FRACTION_BITS)
    if quantized.numel() // shift.numel() > SUM_MAX_TERMS:
        raise ValueError("too many terms for an exact int64 sum")
    total = quantized.to(torch.int64).sum(dim=dim, keepdim=True)
    return total.double() * pow2(-shift)
