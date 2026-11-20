import hashlib
import json

import numpy as np
import torch

SAMPLE_ELEMENTS = 65536


def tensor_bytes(tensor):
    tensor = tensor.detach()
    if tensor.is_floating_point():
        tensor = tensor.float()
    return np.ascontiguousarray(tensor.cpu().numpy())


def sha(tensor):
    return hashlib.sha256(tensor_bytes(tensor).tobytes()).hexdigest()[:16]


class Tracer:
    def __init__(self, trace_steps, sample_dir=None):
        self.trace_steps = trace_steps
        self.sample_dir = sample_dir
        self.step = None
        self.records = []
        self.samples = {}

    def begin(self, step):
        self.step = step if step < self.trace_steps else None

    def end(self):
        if self.step is not None and self.sample_dir is not None:
            np.savez(f"{self.sample_dir}/step{self.step:04d}.npz", **self.samples)
        self.samples = {}
        self.step = None

    def tap(self, name, tensor):
        if self.step is None:
            return
        array = tensor_bytes(tensor)
        self.records.append({
            "step": self.step, "op": name,
            "sha": hashlib.sha256(array.tobytes()).hexdigest()[:16]})
        if self.sample_dir is not None:
            flat = array.reshape(-1)
            stride = max(1, flat.size // SAMPLE_ELEMENTS)
            self.samples[name] = flat[::stride][:SAMPLE_ELEMENTS].copy()

    def write(self, path):
        with open(path, "w") as file:
            for record in self.records:
                file.write(json.dumps(record) + "\n")


ACTIVE = None


def tap(name, tensor):
    if ACTIVE is not None:
        ACTIVE.tap(name, tensor)


def parameter_digest(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(tensor_bytes(parameter.reshape(-1)[:4096]).tobytes())
    return digest.hexdigest()[:16]
