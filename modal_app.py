import json
import subprocess
import time
from pathlib import Path

import modal

UPSTREAM_COMMIT = "b9d678db3a0f726e117c8cab51927975830395ac"
UPSTREAM_DIR = "/opt/tufazip"
DATA_DIR = "/vol/data"
RUNS_DIR = "/vol/runs"
RECORDS_DIR = "/vol/records"
CPU_JOB_CORES = 16

app = modal.App("tunazip")
volume = modal.Volume.from_name("tunazip", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "git")
    .pip_install("torch==2.15.0", "numpy==2.3.4", "tensorboard")
    .run_commands(
        f"git clone https://github.com/Tufalabs/tufazip {UPSTREAM_DIR}",
        f"git -C {UPSTREAM_DIR} checkout {UPSTREAM_COMMIT}")
    .add_local_dir("detzip", "/root/detzip")
    .add_local_dir("scripts", "/root/scripts")
)


def environment():
    import torch
    cuda = torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "torch": torch.__version__, "cuda": torch.version.cuda}


def load_data(name, max_tokens):
    from detzip.codec import load_tokens
    meta = json.loads(Path(f"{DATA_DIR}/{name}.json").read_text())
    tokens = load_tokens(f"{DATA_DIR}/{name}.pp")
    return meta, tokens[:max_tokens] if max_tokens else tokens


def run_prepare(spec):
    from scripts.prepare_data import prepare
    return prepare(DATA_DIR, "/root/detzip/preprocess.c", spec["slices"])


def run_upstream(spec):
    from dataclasses import asdict
    from detzip.codec import Config
    meta, tokens = load_data(spec["data"], spec.get("max_tokens"))
    config = Config(vocab_size=meta["vocab_size"], **spec.get("config", {}))
    arguments = ["--input", f"{DATA_DIR}/{spec['data']}.pp", "--no_coding",
                 "--initial_prior", "", "--work_dir", "/tmp/upstream",
                 "--max_tokens", str(len(tokens))]
    for key, value in asdict(config).items():
        if key not in ("det", "matmul_bits"):
            arguments += [f"--{key}", str(value)]
    started = time.time()
    process = subprocess.run(
        ["python", "/root/scripts/run_upstream.py", UPSTREAM_DIR, *arguments],
        capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stdout[-3000:] + process.stderr[-3000:])
    final = next(line for line in process.stdout.splitlines()
                 if line.startswith("FINAL_RESULT "))
    upstream = json.loads(final.removeprefix("FINAL_RESULT "))
    return {"modeled_bits": upstream["modeled_bits"],
            "modeled_bytes": upstream["modeled_payload_bytes"],
            "coded_symbols": upstream["symbols"],
            "seconds": upstream["elapsed_seconds"],
            "wall_seconds": time.time() - started}


def run_fork(spec):
    import hashlib
    import torch
    from detzip import codec, trace
    meta, tokens = load_data(spec["data"], spec.get("max_tokens"))
    config = codec.Config(vocab_size=meta["vocab_size"], **spec.get("config", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(RUNS_DIR) / spec["run_id"]
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    tracer = None
    if spec.get("trace_steps"):
        tracer = trace.Tracer(
            spec["trace_steps"],
            str(run_dir / "samples") if spec.get("trace_samples") else None)
        trace.ACTIVE = tracer
    options = {"tracer": tracer, "digest_every": spec.get("digest_every", 0)}
    mode = spec.get("mode", "measure")
    payload = Path(RUNS_DIR) / spec.get("payload_run_id", spec["run_id"]) / "payload.bin"
    if mode == "measure":
        result, run = codec.measure(config, tokens, device, **options)
    elif mode == "encode":
        result, run = codec.encode(config, tokens, str(payload), device, **options)
        result["payload_bytes"] = payload.stat().st_size
    else:
        decoded, result, run = codec.decode(str(payload), device, **options)
        result["decoded_matches"] = bool((decoded == tokens).all())
        result["decoded_sha"] = hashlib.sha256(decoded.tobytes()).hexdigest()[:16]
        if not spec.get("max_tokens"):
            result["restores_original_bytes"] = restores_original(spec["data"], decoded)
    if tracer is not None:
        tracer.write(run_dir / "trace.jsonl")
    with open(run_dir / "digests.jsonl", "w") as file:
        for digest in run.digests:
            file.write(json.dumps(digest) + "\n")
    result["raw_bytes"] = meta["raw_bytes"] * len(tokens) // meta["tokens"]
    result["parameters"] = sum(p.numel() for p in run.model.parameters())
    return result


def run_selftest(spec):
    import torch
    from detzip.detops import selftest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"hashes": selftest.op_hashes(device),
            "experiments": selftest.fast_path_experiments(device)}


def run_bench(spec):
    import torch
    from detzip.codec import Config
    from scripts.bench_step import bench
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return bench(Config(**spec["config"]), device, spec.get("online_windows", [192]),
                 spec.get("steps", 10), spec.get("profile", False))


KINDS = {"prepare": run_prepare, "upstream": run_upstream, "fork": run_fork,
         "selftest": run_selftest, "bench": run_bench}


@app.cls(image=image, volumes={"/vol": volume}, timeout=24 * 3600)
class Runner:
    @modal.method()
    def job(self, spec):
        started = time.time()
        result = KINDS[spec["kind"]](spec)
        record = {"spec": spec, "result": result, "environment": environment(),
                  "job_seconds": time.time() - started,
                  "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        Path(RECORDS_DIR).mkdir(parents=True, exist_ok=True)
        Path(f"{RECORDS_DIR}/{modal.current_function_call_id()}.json").write_text(
            json.dumps(record))
        volume.commit()
        return record


def restores_original(name, tokens):
    token_path, restored = Path(f"/tmp/{name}.decoded.pp"), Path(f"/tmp/{name}.restored")
    token_path.write_bytes(tokens.astype(">u2").tobytes())
    subprocess.run([f"{DATA_DIR}/preprocess", "d", f"{DATA_DIR}/{name}.voc", str(token_path),
                    str(restored)], check=True, stdout=subprocess.DEVNULL)
    return restored.read_bytes() == Path(f"{DATA_DIR}/{name}").read_bytes()


def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return commit + ("-dirty" if dirty else "")


@app.local_entrypoint()
def main(specs: str):
    jobs = json.loads(Path(specs).read_text())
    commit = git_state()
    pending = {}
    for index, spec in enumerate(jobs):
        runner = (Runner.with_options(gpu=spec["gpu"]) if spec.get("gpu")
                  else Runner.with_options(cpu=CPU_JOB_CORES))
        pending[index] = runner().job.spawn({**spec, "commit": commit})
    while pending:
        for index, call in list(pending.items()):
            try:
                record = call.get(timeout=2)
            except (TimeoutError, modal.exception.TimeoutError):
                continue
            except Exception as error:
                record = {"spec": jobs[index], "error": repr(error)[:2000]}
            del pending[index]
            print(json.dumps(record), flush=True)
