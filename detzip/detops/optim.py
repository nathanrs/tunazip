import math

import torch

from detzip.detops import fused
from detzip.detops.fixed import exact_sum

CLIP_EPSILON = 1e-6
ADAM_SOURCE = """
template <typename T> void det_adam(T parameter, T grad, T first, T second, T step_size,
                                    T correction, T& new_parameter, T& new_first,
                                    T& new_second) {
    new_first = __dadd_rn(__dmul_rn(first, BETA1), __dmul_rn(grad, ONE_MINUS_BETA1));
    new_second = __dadd_rn(__dmul_rn(second, BETA2),
                           __dmul_rn(__dmul_rn(grad, grad), ONE_MINUS_BETA2));
    T denominator = __dadd_rn(sqrt(new_second) / correction, EPS);
    new_parameter = __dadd_rn(parameter, -__dmul_rn(new_first / denominator, step_size));
}
"""


def clip_gradients(parameters, max_norm):
    device = parameters[0].device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in parameters:
        grad64 = parameter.grad.double()
        total = total + exact_sum(grad64 * grad64, tuple(range(grad64.dim()))).reshape(())
    norm = torch.sqrt(total)
    limit = torch.tensor(max_norm, dtype=torch.float64, device=device)
    scale = torch.clamp(limit / (norm + CLIP_EPSILON), max=1.0)
    for parameter in parameters:
        parameter.grad = (parameter.grad.double() * scale).float()
    return norm


class Adam:
    def __init__(self, parameters, lr, betas, eps):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        device = self.parameters[0].device
        self.first = [torch.zeros_like(p, dtype=torch.float64) for p in self.parameters]
        self.second = [torch.zeros_like(p, dtype=torch.float64) for p in self.parameters]
        self.beta1_power = 1.0
        self.beta2_power = 1.0
        self.step_size = torch.zeros((), dtype=torch.float64, device=device)
        self.correction = torch.zeros((), dtype=torch.float64, device=device)
        self.source = (ADAM_SOURCE
                       .replace("ONE_MINUS_BETA1", fused.literal(1.0 - self.beta1))
                       .replace("ONE_MINUS_BETA2", fused.literal(1.0 - self.beta2))
                       .replace("BETA1", fused.literal(self.beta1))
                       .replace("BETA2", fused.literal(self.beta2))
                       .replace("EPS", fused.literal(eps)))

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    def advance(self):
        self.beta1_power *= self.beta1
        self.beta2_power *= self.beta2
        self.step_size.fill_(self.lr / (1.0 - self.beta1_power))
        self.correction.fill_(math.sqrt(1.0 - self.beta2_power))

    @torch.no_grad()
    def apply(self):
        for parameter, first, second in zip(self.parameters, self.first, self.second):
            if parameter.is_cuda:
                new_parameter, new_first, new_second = fused.kernel(self.source, 3)(
                    parameter, parameter.grad, first, second, self.step_size, self.correction)
                first.copy_(new_first)
                second.copy_(new_second)
                parameter.copy_(new_parameter)
                continue
            grad = parameter.grad.double()
            first.mul_(self.beta1)
            first.add_(grad * (1.0 - self.beta1))
            second.mul_(self.beta2)
            second.add_(grad * grad * (1.0 - self.beta2))
            denominator = torch.sqrt(second) / self.correction + self.eps
            parameter.copy_(
                (parameter.double() - first / denominator * self.step_size).float())

    def step(self):
        self.advance()
        self.apply()
