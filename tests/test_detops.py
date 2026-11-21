from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from detzip import codec
from detzip.detops import fixed, functions, gemm, optim, selftest

GRADIENT_TOLERANCE = 1e-6


def reference_matmul(a, b):
    def quantize(x, axis):
        peak = np.abs(x).max(axis=axis, keepdims=True)
        shift = fixed.MATMUL_BITS - np.frexp(peak)[1]
        return np.rint(x.astype(np.float64) * np.exp2(shift.astype(np.float64))).astype(object), shift

    quantized_a, shift_a = quantize(a, -1)
    quantized_b, shift_b = quantize(b, -2)
    exact = quantized_a.dot(quantized_b)
    out = np.empty(exact.shape, dtype=np.float32)
    for row in range(exact.shape[0]):
        for column in range(exact.shape[1]):
            shift = int(shift_a[row, 0]) + int(shift_b[0, column])
            out[row, column] = np.float32(float(Fraction(int(exact[row, column])) / Fraction(2) ** shift))
    return out


def test_matmul_matches_big_integer_reference():
    torch.manual_seed(0)
    cases = [(torch.randn(7, 33), torch.randn(33, 5)),
             (torch.randn(6, 40) * torch.logspace(-12, 12, 40),
              torch.randn(40, 6) * torch.logspace(8, -8, 6)),
             (torch.zeros(3, 9), torch.randn(9, 4)),
             (torch.full((4, 50), 3.0), torch.full((50, 4), -7.0))]
    for a, b in cases:
        got = gemm.exact_matmul(a, b).numpy()
        want = reference_matmul(a.numpy(), b.numpy())
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def relative_error(a, b):
    return float((a - b).abs().max() / b.abs().max())


def assert_matches_float(det_function, float_function, *inputs):
    det_inputs = [t.clone().requires_grad_() for t in inputs]
    float_inputs = [t.clone().requires_grad_() for t in inputs]
    det_out, float_out = det_function(*det_inputs), float_function(*float_inputs)
    assert relative_error(det_out.detach(), float_out.detach()) < GRADIENT_TOLERANCE
    upstream = torch.randn_like(float_out)
    det_out.backward(upstream)
    float_out.backward(upstream)
    for det_input, float_input in zip(det_inputs, float_inputs):
        assert relative_error(det_input.grad, float_input.grad) < GRADIENT_TOLERANCE


def test_gradients_match_float_autograd():
    torch.manual_seed(0)
    mask = torch.triu(torch.ones(9, 9, dtype=torch.bool), 1)
    targets = torch.randint(0, 50, (16,))
    indices = torch.randint(0, 10, (4, 6))
    assert_matches_float(
        functions.rms_norm,
        lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w,
        torch.randn(4, 9, 32), torch.rand(32) + 0.5)
    assert_matches_float(
        lambda s: functions.masked_softmax(s, mask),
        lambda s: F.softmax(s.masked_fill(mask, -float("inf")), -1),
        torch.randn(2, 3, 9, 9) * 3)
    assert_matches_float(functions.silu, F.silu, torch.randn(5, 7) * 4)
    assert_matches_float(lambda l: functions.cross_entropy(l, targets),
                         lambda l: F.cross_entropy(l, targets), torch.randn(16, 50) * 2)
    assert_matches_float(lambda w: functions.embedding(w, indices),
                         lambda w: F.embedding(indices, w), torch.randn(50, 8))
    key = torch.tensor(12345)
    kept = functions.dropout(torch.ones(6, 40), key, 0.25)
    assert_matches_float(lambda x: functions.dropout(x, key, 0.25),
                         lambda x: x * kept, torch.randn(6, 40))


def test_adam_matches_torch():
    torch.manual_seed(0)
    det_parameter = torch.nn.Parameter(torch.randn(20, 10))
    float_parameter = torch.nn.Parameter(det_parameter.detach().clone())
    det_adam = optim.Adam([det_parameter], betas=(0.9, 0.999), eps=1e-9)
    float_adam = torch.optim.Adam([float_parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-9)
    for _ in range(5):
        grad = torch.randn(20, 10)
        det_parameter.grad, float_parameter.grad = grad.clone(), grad.clone()
        _, scale = optim.gradient_scale([det_parameter], 0.25)
        torch.nn.utils.clip_grad_norm_([float_parameter], 0.25)
        det_adam.advance(1e-3)
        det_adam.apply(scale)
        float_adam.step()
    assert relative_error(det_parameter.data, float_parameter.data) < GRADIENT_TOLERANCE


def test_op_hashes_independent_of_thread_count():
    threads = torch.get_num_threads()
    many = selftest.op_hashes(torch.device("cpu"))
    torch.set_num_threads(1)
    one = selftest.op_hashes(torch.device("cpu"))
    torch.set_num_threads(threads)
    assert many == one


def test_deterministic_round_trip(tmp_path):
    tokens = np.arange(3200, dtype=np.int64) * 7919 % 300
    config = codec.Config(vocab_size=300, n_layer=2, d_model=64, n_head=2, d_inner=176,
                          block_len=1600, retrain_batch_size=8, lr=1e-3,
                          retrain_lr="1e-3", det=True)
    payload = str(tmp_path / "payload.bin")
    _, encoder = codec.encode(config, tokens, payload, torch.device("cpu"),
                              digest_every=10, log=lambda message: None)
    decoded, _, decoder = codec.decode(payload, torch.device("cpu"), digest_every=10,
                                       log=lambda message: None)
    assert (decoded == tokens).all()
    assert encoder.digests == decoder.digests
