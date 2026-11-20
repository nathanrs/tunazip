import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def first_divergence(records_a, records_b):
    for index, (a, b) in enumerate(zip(records_a, records_b)):
        if a != b:
            return index, a, b
    return None


def sample_difference(run_a, run_b, step, op):
    name = f"samples/step{step:04d}.npz"
    if not (Path(run_a) / name).exists() or not (Path(run_b) / name).exists():
        return None
    a = np.load(Path(run_a) / name)[op].astype(np.float64)
    b = np.load(Path(run_b) / name)[op].astype(np.float64)
    return {"max_abs_diff": float(np.abs(a - b).max()),
            "max_abs_value": float(np.abs(a).max()),
            "differing_fraction": float((a != b).mean())}


def compare(run_a, run_b):
    report = {}
    for kind in ("trace", "digests"):
        path_a, path_b = Path(run_a) / f"{kind}.jsonl", Path(run_b) / f"{kind}.jsonl"
        if not path_a.exists() or not path_b.exists():
            continue
        records_a, records_b = load(path_a), load(path_b)
        divergence = first_divergence(records_a, records_b)
        report[kind] = {"records": min(len(records_a), len(records_b))}
        if divergence is None:
            report[kind]["identical"] = True
            continue
        index, a, _ = divergence
        report[kind].update({"identical": False, "first_divergent_record": index,
                             "step": a["step"], "op": a.get("op")})
        if kind == "trace":
            report[kind]["difference"] = sample_difference(run_a, run_b, a["step"], a["op"])
    return report


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))
