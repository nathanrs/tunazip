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
TOP_OPERATIONS = 12


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def online_step(run, tokens, window):
    config = run.config
    inp = torch.from_numpy(tokens[:config.batch_size * window].reshape(
        config.batch_size, window)).to(run.device)
    targets = torch.from_numpy(tokens[:config.batch_size]).to(run.device)
    with run.autocast():
        logits = run.model(inp, slice(-1, None))
    code_logits = logits[:, -1, :].float()
    codec.make_freq_table(code_logits, config.det).cpu()
    run.train_on(code_logits, targets, run.optimizer, False)


def replay_step(run, tokens):
    config = run.config
    batch, length = config.retrain_batch_size, config.retrain_update_len
    inp = torch.from_numpy(tokens[:batch * config.ctx_len].reshape(
        batch, config.ctx_len)).to(run.device)
    targets = torch.from_numpy(tokens[:batch * length]).to(run.device)
    if config.det:
        run.model.next_dropout_keys()
    with run.autocast():
        logits = run.model(inp, slice(config.ctx_len - length, config.ctx_len))
    run.train_on(logits.float().reshape(-1, config.vocab_size), targets,
                 run.retrain_optimizer, False)


def time_steps(step, device, steps):
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
    kernels = [event for event in profiler.events() if event.device_type == DeviceType.CUDA]
    by_device_time = sorted(averages, key=lambda row: -row.self_device_time_total)
    return {
        "operator_calls_per_step": sum(
            row.count for row in averages if row.key.startswith("aten::")) / PROFILED_STEPS,
        "kernels_per_step": len(kernels) / PROFILED_STEPS,
        "kernel_ms_per_step": sum(
            row.self_device_time_total for row in averages) / 1000 / PROFILED_STEPS,
        "top_by_kernel_ms_per_step": {
            row.key: round(row.self_device_time_total / 1000 / PROFILED_STEPS, 3)
            for row in by_device_time[:TOP_OPERATIONS]},
        "top_by_calls_per_step": {
            row.key: row.count / PROFILED_STEPS
            for row in sorted(averages, key=lambda row: -row.count)[:TOP_OPERATIONS]}}


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
