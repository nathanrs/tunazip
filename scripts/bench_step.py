import json
import sys
import time
from dataclasses import replace

import numpy as np
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from detzip import codec

PROFILED_STEPS = 3
WARMUP_STEPS = 3
TOP_OPERATIONS = 12


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def online_step(run, tokens, window):
    config = run.config
    inp = torch.from_numpy(tokens[:config.batch_size * window].reshape(
        config.batch_size, window)).to(run.device)
    targets = torch.from_numpy(tokens[:config.batch_size]).to(run.device)
    run.prepare(run.optimizer, config.lr)
    code_logits, freq = run.graphs.call(
        ("predict", window), (inp,), lambda inp: run.predict(inp, True))
    freq.cpu()
    run.graphs.call(("learn", window), (targets,), lambda targets: (
        run.train_on(code_logits, targets, run.optimizer, False),))


def replay_step(run, tokens):
    config = run.config
    batch, length = config.retrain_batch_size, config.retrain_update_len
    prefix_len = config.ctx_len - length
    inp = torch.from_numpy(tokens[:batch * config.ctx_len].reshape(
        batch, config.ctx_len)).to(run.device)
    targets = torch.from_numpy(tokens[:batch * length].reshape(batch, length)).to(run.device)
    run.prepare(run.retrain_optimizer, config.lr)
    if config.det:
        run.model.next_dropout_keys()
    (loss,) = run.graphs.call(
        ("replay", prefix_len, length), (inp, targets),
        lambda inp, targets: (run.replay_update(inp, targets, prefix_len, length),))
    loss.item()


def time_steps(step, device, steps):
    for _ in range(WARMUP_STEPS):
        step()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        step()
    synchronize(device)
    return (time.perf_counter() - started) * 1000 / steps


def profile_steps(step, device):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as profiler:
        for _ in range(PROFILED_STEPS):
            step()
        synchronize(device)
    averages = profiler.key_averages()
    kernel_ms = {}
    kernels = 0
    for event in profiler.events():
        if event.device_type == DeviceType.CUDA:
            kernels += 1
            kernel_ms[event.name] = (kernel_ms.get(event.name, 0.0)
                                     + event.self_device_time_total / 1000 / PROFILED_STEPS)
    slowest = sorted(kernel_ms.items(), key=lambda item: -item[1])[:TOP_OPERATIONS]
    return {
        "operator_calls_per_step": sum(
            row.count for row in averages if row.key.startswith("aten::")) / PROFILED_STEPS,
        "kernels_per_step": kernels / PROFILED_STEPS,
        "kernel_ms_per_step": sum(kernel_ms.values()),
        "slowest_kernels_ms_per_step": {name[:90]: round(ms, 3) for name, ms in slowest}}


def bench(config, device, online_windows, steps, with_profile):
    tokens = (np.arange(config.batch_size * config.ctx_len, dtype=np.int64) * 7919
              % config.vocab_size)
    results = {}
    for mode, det in (("float", False), ("det", True)):
        run = codec.Codec(replace(config, det=det), device, log=lambda message: None)
        shapes = {f"online_{window}": lambda window=window: online_step(run, tokens, window)
                  for window in online_windows}
        run.model.train()
        shapes["replay"] = lambda: replay_step(run, tokens)
        for name, step in shapes.items():
            run.model.train(name == "replay")
            entry = {"ms_per_step": time_steps(step, device, steps)}
            if with_profile:
                entry["profile"] = profile_steps(step, device)
            results[f"{mode}.{name}"] = entry
    return results


if __name__ == "__main__":
    arguments = json.loads(sys.argv[1])
    config = codec.Config(**arguments["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(bench(config, device, arguments.get("online_windows", [192]),
                           arguments.get("steps", 10), arguments.get("profile", False)),
                     indent=1))
