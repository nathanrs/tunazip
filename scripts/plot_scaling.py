import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
OTHER_RUNS = "#b5b4ad"
BASE_SEED = 1111
BASE_STREAMS = 64


def load_runs(ledger_path, data, block_len):
    runs = []
    for line in Path(ledger_path).read_text().splitlines():
        record = json.loads(line)
        spec = record["spec"]
        config = spec.get("config", {})
        if (spec.get("kind") != "fork" or spec.get("data") != data or "error" in record
                or "max_tokens" in spec
                or config.get("block_len") != block_len or config.get("det")
                or config.get("seed", BASE_SEED) != BASE_SEED
                or config.get("batch_size", BASE_STREAMS) != BASE_STREAMS):
            continue
        result = record["result"]
        runs.append({"size": f'{config["n_layer"]}×{config["d_model"]}',
                     "parameters": result["parameters"], "lr": config["lr"],
                     "bytes": result["modeled_bytes"],
                     "bits_per_byte": result["modeled_bits"] / result["raw_bytes"],
                     "minutes": result["seconds"] / 60})
    return sorted(runs, key=lambda run: (run["parameters"], run["lr"]))


def best_per_size(runs):
    best = {}
    for run in runs:
        if run["size"] not in best or run["bytes"] < best[run["size"]]["bytes"]:
            best[run["size"]] = run
    return list(best.values())


def plot(runs, best, x_key, x_label, log_x, title, note, path):
    figure, axes = plt.subplots(figsize=(7.2, 4.2), dpi=160, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    megabytes = lambda run: run["bytes"] / 1e6
    axes.scatter([run[x_key] for run in runs], [megabytes(run) for run in runs],
                 s=18, color=OTHER_RUNS, linewidths=0, zorder=2)
    axes.plot([run[x_key] for run in best], [megabytes(run) for run in best],
              color=SERIES, linewidth=2, marker="o", markersize=7,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for run in best:
        axes.annotate(run["size"], (run[x_key], megabytes(run)), xytext=(0, -14),
                      textcoords="offset points", ha="center", fontsize=8.5,
                      color=TEXT_PRIMARY)
    if log_x:
        axes.set_xscale("log")
    axes.margins(x=0.06, y=0.14)
    axes.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel("Compressed size (MB)", color=TEXT_SECONDARY, fontsize=9)
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=22)
    axes.text(0, 1.035, f"{note} Blue: best learning rate per size. Grey: other rates.",
              transform=axes.transAxes, color=TEXT_SECONDARY, fontsize=8.5)
    axes.grid(True, which="major", color=GRID, linewidth=0.8)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8.5, length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def table(runs, best):
    smallest = min(run["bytes"] for run in best)
    winners = {id(run) for run in best}
    lines = ["| Size | Parameters | Learning rate | Bytes | Bits per byte | Behind best | Minutes |",
             "|---|---|---|---|---|---|---|"]
    for run in runs:
        mark = "**" if id(run) in winners else ""
        lines.append(f'| {run["size"]} | {run["parameters"] / 1e6:.1f}M | {run["lr"]:g} '
                     f'| {mark}{run["bytes"]:,.0f}{mark} | {run["bits_per_byte"]:.4f} '
                     f'| +{(run["bytes"] / smallest - 1) * 100:.1f}% | {run["minutes"]:.0f} |')
    return "\n".join(lines)


if __name__ == "__main__":
    ledger_path, data, block_len, out_prefix = sys.argv[1:5]
    runs = load_runs(ledger_path, data, int(block_len))
    note = f"{int(block_len):,}-token blocks."
    best = best_per_size(runs)
    megabytes = data.split("-")[1].replace("MB", " MB")
    plot(runs, best, "parameters", "Parameters (log scale)", True,
         f"Compressed size of the first {megabytes} of enwik8 by model size", note,
         f"{out_prefix}-parameters.png")
    plot(runs, best, "minutes", "Wall time on one H100 (minutes)", False,
         f"Compressed size of the first {megabytes} of enwik8 by wall time", note,
         f"{out_prefix}-minutes.png")
    print(table(runs, best))
