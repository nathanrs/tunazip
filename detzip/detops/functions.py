import torch

from detzip.detops.elementwise import exp64, sigmoid64
from detzip.detops.fixed import SUM_FRACTION_BITS, exact_sum, pow2, quantize

RMS_EPSILON = 1e-5


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x64, weight64 = x.double(), weight.double()
        mean_square = exact_sum(x64 * x64, -1) * (1.0 / x.shape[-1])
        inverse_rms = 1.0 / torch.sqrt(mean_square + RMS_EPSILON)
        ctx.save_for_backward(x, weight, inverse_rms)
        return (x64 * inverse_rms * weight64).float()

    @staticmethod
    def backward(ctx, grad):
        x, weight, inverse_rms = ctx.saved_tensors
        x64, grad64 = x.double(), grad.double()
        scaled = grad64 * weight.double()
        projection = exact_sum(scaled * x64, -1) * (1.0 / x.shape[-1])
        grad_x = scaled * inverse_rms - x64 * (inverse_rms * inverse_rms * inverse_rms) * projection
        leading = tuple(range(x.dim() - 1))
        grad_weight = exact_sum(grad64 * x64 * inverse_rms, leading).reshape(weight.shape)
        return grad_x.float(), grad_weight.float()


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
        quantized, shift = quantize(rows, (0, 1), SUM_FRACTION_BITS)
        total = torch.zeros(ctx.weight_shape, dtype=torch.int64, device=grad.device)
        total.index_add_(0, indices.reshape(-1), quantized.to(torch.int64))
        return (total.double() * pow2(-shift)).float(), None


def embedding(weight, indices):
    return EmbeddingFunction.apply(weight, indices)
