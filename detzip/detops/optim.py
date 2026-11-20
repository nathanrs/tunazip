import math

import torch

from detzip.detops.fixed import exact_sum

CLIP_EPSILON = 1e-6


def clip_gradients(parameters, max_norm):
    total = 0.0
    for parameter in parameters:
        grad64 = parameter.grad.double()
        total += exact_sum(grad64 * grad64, tuple(range(grad64.dim()))).item()
    norm = math.sqrt(total)
    scale = min(1.0, max_norm / (norm + CLIP_EPSILON))
    if scale < 1.0:
        for parameter in parameters:
            parameter.grad = (parameter.grad.double() * scale).float()
    return norm


class Adam:
    def __init__(self, parameters, lr, betas, eps):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.first = [torch.zeros_like(p, dtype=torch.float64) for p in self.parameters]
        self.second = [torch.zeros_like(p, dtype=torch.float64) for p in self.parameters]
        self.beta1_power = 1.0
        self.beta2_power = 1.0

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        self.beta1_power *= self.beta1
        self.beta2_power *= self.beta2
        step_size = self.lr / (1.0 - self.beta1_power)
        correction = math.sqrt(1.0 - self.beta2_power)
        for parameter, first, second in zip(self.parameters, self.first, self.second):
            grad = parameter.grad.double()
            first.mul_(self.beta1)
            first.add_(grad * (1.0 - self.beta1))
            second.mul_(self.beta2)
            second.add_(grad * grad * (1.0 - self.beta2))
            denominator = torch.sqrt(second) / correction + self.eps
            parameter.copy_((parameter.double() - first / denominator * step_size).float())
