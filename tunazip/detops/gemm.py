from dataclasses import dataclass

import torch

from tunazip.detops.fixed import grid_round

FLOAT64_EXACT_INTEGER_LIMIT = 2 ** 53


@dataclass
class Bits:
    activation: int = 16
    weight: int = 16
    gradient: int = 16


BITS = Bits()


def exact_matmul(a, b, bits_a, bits_b):
    if a.shape[:-2] != b.shape[:-2]:
        raise ValueError(f"batch shapes differ: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.shape[-1] * 2 ** (bits_a + bits_b) > FLOAT64_EXACT_INTEGER_LIMIT:
        raise ValueError(f"inner dimension {a.shape[-1]} overflows the exact float64 range")
    return torch.matmul(grid_round(a, -1, bits_a), grid_round(b, -2, bits_b)).float()


class MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, right_bits):
        ctx.save_for_backward(a, b)
        ctx.right_bits = right_bits
        return exact_matmul(a, b, BITS.activation, right_bits)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return (exact_matmul(grad, b.transpose(-1, -2), BITS.gradient, ctx.right_bits),
                exact_matmul(a.transpose(-1, -2), grad, BITS.activation, BITS.gradient),
                None)


def matmul(a, b):
    return MatmulFunction.apply(a, b, BITS.activation)


def linear(x, weight):
    flat = MatmulFunction.apply(x.reshape(-1, x.shape[-1]), weight.t(), BITS.weight)
    return flat.reshape(*x.shape[:-1], weight.shape[0])
