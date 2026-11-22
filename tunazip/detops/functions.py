import torch

from tunazip.detops import fused, rng
from tunazip.detops.elementwise import exp64, sigmoid64
from tunazip.detops.fixed import (SUM_FRACTION_BITS, exact_sum, product, quantize,
                                 scale_twice)

RMS_EPSILON = 1e-5


INVERSE_RMS_SOURCE = """
template <typename T> T inverse_rms(T square_sum) {
    return 1.0 / sqrt(__dadd_rn(__dmul_rn(square_sum, INVERSE_WIDTH), EPSILON));
}
"""
RMS_BACKWARD_SOURCE = """
template <typename T> T rms_backward(T grad, T weight, T x, T inverse_rms, T projection_sum) {
    T cube = __dmul_rn(__dmul_rn(inverse_rms, inverse_rms), inverse_rms);
    T projection = __dmul_rn(projection_sum, INVERSE_WIDTH);
    return __dadd_rn(__dmul_rn(__dmul_rn(grad, weight), inverse_rms),
                     -__dmul_rn(__dmul_rn(x, cube), projection));
}
"""


def width_source(source, width):
    return (source.replace("INVERSE_WIDTH", fused.literal(1.0 / width))
            .replace("EPSILON", fused.literal(RMS_EPSILON)))


def inverse_rms_of(square_sum, width):
    if square_sum.is_cuda:
        return fused.kernel(width_source(INVERSE_RMS_SOURCE, width))(square_sum)
    return 1.0 / torch.sqrt(square_sum * (1.0 / width) + RMS_EPSILON)


def rms_input_gradient(grad, weight, x64, inverse_rms, projection_sum, width):
    if grad.is_cuda:
        return fused.kernel(width_source(RMS_BACKWARD_SOURCE, width))(
            grad, weight, x64, inverse_rms, projection_sum)
    scaled = grad.double() * weight.double()
    cube = inverse_rms * inverse_rms * inverse_rms
    return scaled * inverse_rms - x64 * cube * (projection_sum * (1.0 / width))


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x64 = x.double()
        inverse_rms = inverse_rms_of(exact_sum(product(x64, x64), -1), x.shape[-1])
        ctx.save_for_backward(x, weight, inverse_rms)
        return scale_twice(x64, inverse_rms, weight).float()

    @staticmethod
    def backward(ctx, grad):
        x, weight, inverse_rms = ctx.saved_tensors
        x64 = x.double()
        projection_sum = exact_sum(scale_twice(x64, grad, weight), -1)
        grad_x = rms_input_gradient(grad, weight, x64, inverse_rms, projection_sum, x.shape[-1])
        leading = tuple(range(x.dim() - 1))
        grad_weight = exact_sum(scale_twice(x64, grad, inverse_rms), leading)
        return grad_x.float(), grad_weight.reshape(weight.shape).float()


def rms_norm(x, weight):
    return RMSNormFunction.apply(x, weight)


def softmax64(scores, mask=None):
    if mask is not None:
        scores = scores.masked_fill(mask, -float("inf"))
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    exponentials = exp64(shifted.double())
    if mask is not None:
        exponentials = exponentials.masked_fill(mask, 0.0)
    return exponentials / exact_sum(exponentials, -1)


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mask):
        probabilities = softmax64(scores, mask)
        ctx.save_for_backward(probabilities)
        return probabilities.float()

    @staticmethod
    def backward(ctx, grad):
        (probabilities,) = ctx.saved_tensors
        weighted = grad.double() * probabilities
        return (weighted - probabilities * exact_sum(weighted, -1)).float(), None


def masked_softmax(scores, mask):
    return SoftmaxFunction.apply(scores, mask)


class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid = sigmoid64(x.double())
        ctx.save_for_backward(x, sigmoid)
        return (x.double() * sigmoid).float()

    @staticmethod
    def backward(ctx, grad):
        x, sigmoid = ctx.saved_tensors
        slope = sigmoid + x.double() * sigmoid * (1.0 - sigmoid)
        return (grad.double() * slope).float()


def silu(x):
    return SiLUFunction.apply(x)


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        probabilities = softmax64(logits)
        ctx.save_for_backward(probabilities, targets)
        picked = probabilities.gather(-1, targets[:, None])
        return -torch.log(picked).mean().float()

    @staticmethod
    def backward(ctx, grad):
        probabilities, targets = ctx.saved_tensors
        one_hot = torch.zeros_like(probabilities).scatter_(-1, targets[:, None], 1.0)
        scale = 1.0 / probabilities.shape[0]
        return ((probabilities - one_hot) * scale * grad.double()).float(), None


def cross_entropy(logits, targets):
    return CrossEntropyFunction.apply(logits, targets)


class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.save_for_backward(indices)
        ctx.weight_shape = weight.shape
        return weight[indices]

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        rows = grad.reshape(-1, grad.shape[-1])
        quantized, inverse = quantize(rows, (0, 1), SUM_FRACTION_BITS)
        total = torch.zeros(ctx.weight_shape, dtype=torch.int64, device=grad.device)
        total.index_add_(0, indices.reshape(-1), quantized.to(torch.int64))
        return product(total, inverse).float(), None


def embedding(weight, indices):
    return EmbeddingFunction.apply(weight, indices)


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, key, drop_probability):
        ctx.key, ctx.drop_probability = key, drop_probability
        return rng.dropout(x, key, drop_probability)

    @staticmethod
    def backward(ctx, grad):
        return rng.dropout(grad, ctx.key, ctx.drop_probability), None, None


def dropout(x, key, drop_probability):
    return DropoutFunction.apply(x, key, drop_probability)
