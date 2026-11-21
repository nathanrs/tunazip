from decimal import Context, Decimal, localcontext
from fractions import Fraction
from math import factorial

import torch

from detzip.detops import fused
from detzip.detops.fixed import pow2

INV_LN2 = float.fromhex("0x1.71547652b82fep+0")
LN2_HI = float.fromhex("0x1.62e42fee00000p-1")
LN2_LO = float.fromhex("0x1.a39ef35793c76p-33")
EXP_COEFFICIENTS = [float(Fraction(1, factorial(n))) for n in range(14)]
EXP_INPUT_LIMIT = 700.0
HOST_CONTEXT = Context(prec=60)
HOST_EPSILON = Decimal(10) ** -55


EXP_STATEMENTS = f"""
    x = x < -{fused.literal(EXP_INPUT_LIMIT)} ? -{fused.literal(EXP_INPUT_LIMIT)}
        : (x > {fused.literal(EXP_INPUT_LIMIT)} ? {fused.literal(EXP_INPUT_LIMIT)} : x);
    T k = nearbyint(__dmul_rn(x, {fused.literal(INV_LN2)}));
    T r = __dadd_rn(x, -__dmul_rn(k, {fused.literal(LN2_HI)}));
    r = __dadd_rn(r, -__dmul_rn(k, {fused.literal(LN2_LO)}));
    T p = {fused.literal(EXP_COEFFICIENTS[-1])};
""" + "".join(f"    p = __dadd_rn(__dmul_rn(p, r), {fused.literal(coefficient)});\n"
              for coefficient in reversed(EXP_COEFFICIENTS[:-1])) + """
    T e = __dmul_rn(p, __longlong_as_double(((long long)k + 1023LL) << 52));
"""
EXP_SOURCE = "template <typename T> T det_exp(T x) {" + EXP_STATEMENTS + "    return e;\n}"
SIGMOID_SOURCE = ("template <typename T> T det_sigmoid(T v) {\n    T x = -v;" + EXP_STATEMENTS
                  + "    return 1.0 / __dadd_rn(1.0, e);\n}")


def exp64(x):
    if x.is_cuda:
        return fused.kernel(EXP_SOURCE)(x)
    x = x.clamp(-EXP_INPUT_LIMIT, EXP_INPUT_LIMIT)
    k = torch.round(x * INV_LN2)
    r = x - k * LN2_HI
    r = r - k * LN2_LO
    p = torch.full_like(r, EXP_COEFFICIENTS[-1])
    for coefficient in reversed(EXP_COEFFICIENTS[:-1]):
        p = p * r
        p = p + coefficient
    return p * pow2(k)


def sigmoid64(x):
    if x.is_cuda:
        return fused.kernel(SIGMOID_SOURCE)(x)
    return 1.0 / (1.0 + exp64(-x))


def _arctan_inverse(n):
    total = term = Decimal(1) / n
    k = 1
    while abs(term) > HOST_EPSILON:
        term = -term / (n * n)
        total += term / (2 * k + 1)
        k += 1
    return total


def _pi():
    return 16 * _arctan_inverse(5) - 4 * _arctan_inverse(239)


def _cos_sin(angle, pi):
    angle = angle % (2 * pi)
    cos = sin = Decimal(0)
    term = Decimal(1)
    n = 0
    while abs(term) > HOST_EPSILON or n < 20:
        if n % 2 == 0:
            cos += term if n % 4 == 0 else -term
        else:
            sin += term if n % 4 == 1 else -term
        n += 1
        term = term * angle / n
    return cos, sin


def host_cos_of_pi_times(fraction):
    with localcontext(HOST_CONTEXT):
        pi = _pi()
        return float(_cos_sin(pi * Decimal(fraction), pi)[0])


def rope_tables(seq_len, head_dim, base=10000):
    with localcontext(HOST_CONTEXT):
        pi = _pi()
        log_base = Decimal(base).ln()
        cos_rows = [[0.0] * head_dim for _ in range(seq_len)]
        sin_rows = [[0.0] * head_dim for _ in range(seq_len)]
        for pair in range(head_dim // 2):
            inverse_frequency = (-log_base * Decimal(2 * pair) / head_dim).exp()
            for position in range(seq_len):
                cos, sin = _cos_sin(position * inverse_frequency, pi)
                for column in (pair, pair + head_dim // 2):
                    cos_rows[position][column] = float(cos)
                    sin_rows[position][column] = float(sin)
    return (torch.tensor(cos_rows, dtype=torch.float64).float(),
            torch.tensor(sin_rows, dtype=torch.float64).float())
