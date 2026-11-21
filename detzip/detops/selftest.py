import torch

from detzip import trace
from detzip.detops import functions, gemm, optim, rng
from detzip.detops.elementwise import exp64, host_cos_of_pi_times, rope_tables, sigmoid64
from detzip.detops.fixed import exact_sum, pow2


def normal(key, shape, device, std=1.0):
    return rng.normal(rng.stream_key(99, key), shape, std).to(device)


def wide_range(key, shape, device):
    exponents = rng.uniform_integers(rng.stream_key(98, key), shape[-1], torch.device("cpu"))
    scale = pow2(exponents % 61 - 30).float()
    return (normal(key, shape, torch.device("cpu")) * scale).to(device)


def op_hashes(device):
    hashes = {}

    def record(name, tensor):
        hashes[name] = trace.sha(tensor)

    a, b = normal(1, (96, 384), device), normal(2, (384, 160), device)
    record("matmul", gemm.exact_matmul(a, b))
    record("matmul_wide_range", gemm.exact_matmul(
        wide_range(3, (96, 384), device), wide_range(4, (384, 160), device)))
    record("matmul_batched", gemm.exact_matmul(
        normal(5, (4, 2, 48, 32), device), normal(6, (4, 2, 32, 48), device)))
    record("matmul_long_inner", gemm.exact_matmul(
        normal(7, (16, 12288), device), normal(8, (12288, 16), device)))

    x = normal(9, (8, 48, 256), device).requires_grad_()
    weight = normal(10, (512, 256), device, 0.02).requires_grad_()
    out = gemm.linear(x, weight)
    out.backward(normal(11, tuple(out.shape), device))
    record("linear", out)
    record("linear_grad_x", x.grad)
    record("linear_grad_w", weight.grad)

    record("exact_sum", exact_sum(wide_range(12, (64, 4096), device), -1))
    sweep = (torch.arange(2_000_001, dtype=torch.int64, device=device) - 1_000_000).double() * 6e-5
    record("exp64", exp64(sweep))
    record("sigmoid64", sigmoid64(sweep))
    record("sqrt_div", 1.0 / torch.sqrt(sweep * sweep + 1e-5))
    record("reciprocal_scale", sweep * (1.0 / 384))
    record("tensor_div", torch.sqrt(sweep * sweep + 1e-5)
           / torch.tensor(0.7310585786300049, dtype=torch.float64, device=device))

    x = normal(13, (8, 48, 384), device).requires_grad_()
    gain = (normal(14, (384,), device, 0.1) + 1.0).requires_grad_()
    out = functions.rms_norm(x, gain)
    out.backward(normal(15, tuple(out.shape), device))
    record("rms_norm", out)
    record("rms_norm_grad_x", x.grad)
    record("rms_norm_grad_w", gain.grad)

    scores = normal(16, (4, 2, 48, 48), device, 3.0).requires_grad_()
    mask = torch.triu(torch.ones(48, 48, dtype=torch.bool, device=device), diagonal=1)
    out = functions.masked_softmax(scores, mask)
    out.backward(normal(17, tuple(out.shape), device))
    record("softmax", out)
    record("softmax64", functions.softmax64(scores.detach(), mask))
    record("softmax_grad", scores.grad)

    x = normal(18, (8, 48, 256), device, 4.0).requires_grad_()
    out = functions.silu(x)
    out.backward(normal(19, tuple(out.shape), device))
    record("silu", out)
    record("silu_grad", x.grad)

    logits = normal(20, (48, 4098), device, 2.0).requires_grad_()
    targets = (rng.uniform_integers(rng.stream_key(97), 48, device) % 4098)
    functions.cross_entropy(logits, targets).backward()
    record("cross_entropy_grad", logits.grad)
    record("freq_softmax", torch.round(functions.softmax64(logits.detach()) * 10_000_000))

    table = normal(21, (4098, 256), device, 0.02).requires_grad_()
    indices = (rng.uniform_integers(rng.stream_key(96), 64 * 48, device) % 300).reshape(64, 48)
    out = functions.embedding(table, indices)
    out.backward(wide_range(22, tuple(out.shape), device))
    record("embedding_grad", table.grad)

    parameter = torch.nn.Parameter(normal(23, (512, 256), device, 0.02))
    adam = optim.Adam([parameter], lr=4e-5, betas=(0.0, 0.9999), eps=1e-9)
    for step in range(3):
        parameter.grad = wide_range(24 + step, (512, 256), device)
        norm, scale = optim.gradient_scale([parameter], 0.25)
        adam.step(scale)
    record("adam", parameter)
    record("adam_second", adam.second[0])
    hashes["clip_norm"] = float(norm).hex()

    record("dropout", rng.dropout(
        normal(27, (64, 48, 256), device),
        torch.tensor(rng.stream_key(95), dtype=torch.int64, device=device), 0.25))
    cos, sin = rope_tables(48, 32)
    record("rope_cos", cos)
    record("rope_sin", sin)
    hashes["host_cos"] = float(host_cos_of_pi_times(0.3721)).hex()
    return hashes


def integer_reference(a, b):
    return (a.cpu().to(torch.int64) @ b.cpu().to(torch.int64))


def fast_path_experiments(device):
    report = {}
    if device.type != "cuda":
        return report
    torch.manual_seed(0)
    cases = {
        "random": (torch.randint(-127, 128, (256, 1024)), torch.randint(-127, 128, (1024, 512))),
        "worst": (torch.full((256, 1024), 127), torch.full((1024, 512), 127)),
        "alternating": (torch.full((256, 1024), 127),
                        torch.tensor([127, -127]).repeat(512)[:, None].expand(1024, 512).clone()),
    }
    for name, (a, b) in cases.items():
        want = integer_reference(a, b)
        a_gpu, b_gpu = a.to(device), b.to(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        report[f"fp32.{name}"] = bool((a_gpu.float() @ b_gpu.float()).cpu().double().eq(want.double()).all())
        torch.backends.cuda.matmul.allow_tf32 = True
        report[f"tf32.{name}"] = bool((a_gpu.float() @ b_gpu.float()).cpu().double().eq(want.double()).all())
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            got = torch._int_mm(a_gpu.to(torch.int8), b_gpu.to(torch.int8))
            report[f"int_mm.{name}"] = bool(got.cpu().to(torch.int64).eq(want).all())
        except Exception as error:
            report[f"int_mm.{name}"] = repr(error)[:200]
    return report
