# tunazip

tunazip compresses text with a transformer that trains while it compresses, and its files decode byte-exactly on a different GPU. It is a fork of [tufazip](https://github.com/Tufalabs/tufazip).

The decoder has to repeat the encoder's arithmetic bit for bit. Floating-point results differ between GPU models, so a float-mode file decodes only on the GPU type that encoded it. tunazip adds a deterministic mode that gives the same bits on every device tested.

## Results

- enwik8 (100 MB) compresses to 16.4 MB, which is 1.31 bits per byte. zip gives 36.4 MB, and xz, a stronger standard compressor, gives 24.8 MB.
- Deterministic mode costs no measurable bytes. Its size lands within the run-to-run spread of float mode.
- Deterministic mode is about 3 times slower: 17.9 hours against 5.5 to 6.2 hours on an H100.
- A file encoded on an H100 decodes byte-exactly on a B200, A100, L40S, A10 and T4. The full 100 MB file was decoded on the B200.
- A float-mode file fails on all five.

The measurements are in [results/](results/summary.md).

## How it works

- Every matmul operand is rounded to 16 significant bits, and the matmul runs in float64, where the result is exact.
- Sums run in integers.
- The CUDA kernels never fuse a multiply and an add, and `exp` is a fixed polynomial.

## Usage

Jobs run on [Modal](https://modal.com). You need `uv` and a Modal account.

```bash
uv sync
```

```bash
uv run modal setup
```

Each job is one entry in a JSON file of specs, run with:

```bash
uv run modal run modal_app.py --specs <file>
```

Prepare the first 1 MB of enwik8:

```json
[{"kind": "prepare", "slices": ["enwik8-1MB"]}]
```

Encode it on an H100 in deterministic mode:

```json
[{"kind": "fork", "gpu": "H100!", "data": "enwik8-1MB", "mode": "encode",
  "run_id": "demo-encode",
  "config": {"n_layer": 4, "d_model": 384, "n_head": 3, "d_inner": 1056,
             "lr": 0.0004, "block_len": 470, "retrain_len": 32890,
             "retrain_lr": "0.0016,12,0.0016,87,0.00079,153,0.0004,1796,0.00013",
             "det": true, "seed": 1111}}]
```

Decode it on a B200 and compare with the original bytes:

```json
[{"kind": "fork", "gpu": "B200", "data": "enwik8-1MB", "mode": "decode",
  "run_id": "demo-decode", "payload_run_id": "demo-encode"}]
```

A decode takes about as long as the encode, 7 minutes each here. Every run used torch 2.15.0, and other versions are untested.

## License and credits

tunazip is MIT-licensed. See [LICENSE](LICENSE).

It derives from tufazip, Fabrice Bellard's NNCP preprocessor and Project Nayuki's reference arithmetic coder, all MIT-licensed. See [NOTICE](NOTICE).
