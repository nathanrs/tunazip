import torch

from detzip.detops import fixed
from detzip.detops.fixed import quantize, scale_twice

FLOAT64_EXACT_INTEGER_LIMIT = 2 ** 53


def exact_matmul(a, b):
    if a.shape[:-2] != b.shape[:-2]:
        raise ValueError(f"batch shapes differ: {tuple(a.shape)} @ {tuple(b.shape)}")
    bits = fixed.MATMUL_BITS
    if a.shape[-1] * 2 ** (2 * bits) > FLOAT64_EXACT_INTEGER_LIMIT:
        raise ValueError(f"inner dimension {a.shape[-1]} overflows the exact float64 range")
    quantized_a, inverse_a = quantize(a, -1, bits)
    quantized_b, inverse_b = quantize(b, -2, bits)
    return scale_twice(torch.matmul(quantized_a, quantized_b), inverse_a, inverse_b).float()


class MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return exact_matmul(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return (exact_matmul(grad, b.transpose(-1, -2)),
                exact_matmul(a.transpose(-1, -2), grad))


def matmul(a, b):
    return MatmulFunction.apply(a, b)


def linear(x, weight):
    flat = matmul(x.reshape(-1, x.shape[-1]), weight.t())
    return flat.reshape(*x.shape[:-1], weight.shape[0])
