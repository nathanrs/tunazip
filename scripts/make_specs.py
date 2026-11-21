import json
import sys

UPSTREAM_TOKENS = 285_202_830
UPSTREAM_ONLINE_LR = 4e-5
UPSTREAM_BLOCK_LEN = 500_000
UPSTREAM_RETRAIN_LEN = 35_000_000
UPSTREAM_RETRAIN_SCHEDULE = [(0, 1.6e-4), (13000, 1.6e-4), (93000, 7.9e-5),
                             (163000, 4.0e-5), (1911300, 1.3e-5)]
SLICE_TOKENS = {"enwik8-1MB": 268_011, "enwik8-10MB": 3_065_672,
                "enwik8-100MB": 31_330_083}
HEAD_DIM = 128
INNER_RATIO = 2.75
SIZES = {(2, 256): [2e-4, 5e-4, 1.2e-3], (4, 384): [1.5e-4, 4e-4, 1e-3],
         (6, 512): [1e-4, 2.5e-4, 6e-4], (8, 640): [8e-5, 2e-4, 5e-4],
         (12, 768): [6e-5, 1.5e-4, 4e-4], (12, 1024): [4e-5, 1e-4, 2.5e-4]}


def scaled_recipe(tokens, lr):
    scale = tokens / UPSTREAM_TOKENS
    lr_scale = lr / UPSTREAM_ONLINE_LR
    knots = [f"{UPSTREAM_RETRAIN_SCHEDULE[0][1] * lr_scale:.4g}"]
    previous_step = 0
    for step, rate in UPSTREAM_RETRAIN_SCHEDULE[1:]:
        previous_step = max(previous_step + 1, round(step * scale))
        knots += [str(previous_step), f"{rate * lr_scale:.4g}"]
    return {"lr": lr, "block_len": round(UPSTREAM_BLOCK_LEN * scale),
            "retrain_len": round(UPSTREAM_RETRAIN_LEN * scale),
            "retrain_lr": ",".join(knots)}


def model_shape(n_layer, d_model):
    return {"n_layer": n_layer, "d_model": d_model, "n_head": d_model // HEAD_DIM,
            "d_inner": round(d_model * INNER_RATIO)}


def scaling_sweep(data, gpu):
    specs = []
    for (n_layer, d_model), rates in SIZES.items():
        for lr in rates:
            specs.append({
                "kind": "fork", "gpu": gpu, "data": data,
                "run_id": f"scale-{data}-{n_layer}x{d_model}-lr{lr:g}",
                "config": {**model_shape(n_layer, d_model),
                           **scaled_recipe(SLICE_TOKENS[data], lr)}})
    return specs


if __name__ == "__main__":
    json.dump(scaling_sweep(sys.argv[1], sys.argv[2]), sys.stdout, indent=1)
