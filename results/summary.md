# detzip results

detzip is a fork of tufazip whose compressed files decode byte-exactly on a different GPU. Details are in [determinism.md](determinism.md) and [scaling.md](scaling.md).

## Determinism costs 0.10% of the bytes and 2.9 to 3.3 times the time on enwik8

4×384 (10.4M parameters), 64 streams, random initial weights, one H100, all 100,000,000 bytes of enwik8.

| Mode | Seed | Modeled bytes | Bits per byte | Minutes |
|---|---|---|---|---|
| Float bf16 | 1111 | 16,362,011 | 1.3090 | 329 |
| Float bf16 | 2222 | 16,394,003 | 1.3115 | 372 |
| Deterministic, 16 bits | 1111 | 16,394,464 | 1.3116 | 1,076 |

- Bytes lost to determinism: 16,457 against the float mean of 16,378,007, which is 0.10%.
- The two float seeds differ by 0.20%, so the loss is inside seed noise. The deterministic run has one seed.
- Slowdown from determinism: 3.27 times against float seed 1111 and 2.89 times against float seed 2222. Float wall time varies with the host CPU.
- The deterministic file is 16,424,127 bytes with its dictionary, 1.3139 bits per byte. xz -9e gives 24,831,656 bytes.
- On 10 MB with two seeds per mode, the deterministic mean is 0.18% larger than float with long blocks and 0.23% smaller with short blocks. Both are inside seed noise.
- On 10 MB with full contexts the slowdown is 3.0 to 3.4 times.

## A float-mode file decodes only on the GPU type that encoded it

- A 1 MB file encoded in float mode on an H100 decodes on a second H100.
- On B200, A100, L40S, A10 and T4 the decoder goes wrong within the first three tokens and gets over 99.9% of the 268,011 tokens wrong.

## A file encoded on an H100 decodes byte-exactly on five other GPU types

| File | Decoders that restored the original bytes |
|---|---|
| 1 MB, 275,620-byte payload | B200, A100, L40S, A10, T4 |
| 10 MB, 2,101,694-byte payload | B200, A100 |
| 100 MB, 16,395,046-byte payload | B200 decode is running |

- The six GPU types span five generations: Turing, Ampere, Ada, Hopper and Blackwell.
- 31 operation hashes are identical on the six GPU types, a Linux CPU and a macOS CPU.
- A 16,000-token training run ends with the same parameter hash on the six GPU types and the CPU.
- A full decode on a CPU did not finish. The preemptible container restarted seven times, and the job has no checkpoints.

## How the deterministic mode gets exact results

- Every matmul operand is rounded to 16 significant bits per row or column, on a power-of-two grid.
- The matmul runs in float64. The result is exact while the inner dimension times 2^(bits of one operand + bits of the other) stays within 2^53.
- Sums run in int64.
- `exp` is a fixed polynomial evaluated with separate multiplies and adds, and sigmoid is built on it.
- Random numbers come from a splitmix64 counter. Adam is written out with explicit operations.
- The CUDA kernels use `__dmul_rn` and `__dadd_rn`, so the compiler cannot fuse a multiply and an add.

## Fused kernels and a CUDA graph bring the step from 140 ms to 37 ms

- Fused kernels cut the operations per step from about 20,000 to about 6,000.
- Each step shape is captured once as a CUDA graph and replayed.
- A 4×384 replay step takes 37.1 ms deterministic and 15.7 ms float on an H100.
- A 4×384 online step over 192 tokens takes 62.8 ms deterministic and 16.9 ms float.
- At 12×768 the deterministic replay step takes 4.4 times the float step.

## Gradients need 16 bits; activations and weights do not

Activation / weight / gradient bits, 4×384, 10 MB, against 16/16/16.

| Bits | Short blocks | Long blocks |
|---|---|---|
| 8 / 8 / 16 | +0.46% | −0.41% |
| 12 / 12 / 16 | −0.11% | +0.45% |
| 16 / 16 / 12 | +2.95% | +5.21% |
| 8 / 8 / 12 | +7.9% and +11.6% | |
| 16 / 16 / 20 | | +0.27% |

- 12-bit gradients cost 3% to 5% on 10 MB and nothing on 1 MB. A bit width has to be checked at the size of the file.
- 8-bit activations and weights stay within seed noise of 16 bits.
- 20-bit gradients do not beat 16-bit gradients on 10 MB.
- On 100 MB a 22-bit-gradient run was 0.41% ahead of the 16-bit run when it was stopped at 21.09M tokens. One seed each, and two float seeds end 0.20% apart.
- Lower bit widths buy no speed on an H100. A float32 forward matmul moved the replay step from 37.1 ms to 36.5 ms.

## On 100 MB, 6×512 is 3.0% smaller than 4×384 and matches 12×768

| Size | Parameters | Bytes | Bits per byte | Minutes | GPU |
|---|---|---|---|---|---|
| 4×384 | 10.4M | 16,362,011 | 1.3090 | 329 | H100 |
| 6×512 | 23.5M | 15,867,295 | 1.2694 | 543 | H100 |
| 12×768 | 93.0M | 15,882,829 | 1.2706 | 909 | B200 |

- Float mode, seed 1111. A second seed of 4×384 gives 16,394,003 bytes, 0.20% more.
- 4×384 is 0.4% behind 6×512 after 1.1M tokens and 3.1% behind after 31.3M tokens.
- The learning rates came from the short-block 10 MB sweep. The long-block 10 MB sweep prefers a quarter of that rate for 4×384 and 6×512.

## On 10 MB with long blocks, sizes from 4×384 to 12×1024 are within 1.0%

Float mode, 64 streams, best of three to five learning rates, mean of two seeds.

| Size | Parameters | Learning rate | Mean bytes | Behind best | Minutes |
|---|---|---|---|---|---|
| 2×256 | 3.7M | 6e-4 | 2,120,634 | +5.0% | 13 |
| 4×384 | 10.4M | 2.5e-4 | 2,026,194 | +0.3% | 23 |
| 6×512 | 23.5M | 3e-4 | 2,019,821 | best | 36 |
| 8×640 | 45.4M | 5e-4 | 2,025,016 | +0.3% | 71 |
| 12×768 | 93.0M | 8e-4 | 2,033,830 | +0.7% | 100 |
| 12×1024 | 162.6M | 5e-4 | 2,040,506 | +1.0% | 131 |

- The upstream 12×1024 architecture is 1.0% behind 6×512 and takes 3.6 times as long.
- Two seeds of one size differ by 0.12% to 0.32%.
- Every run takes 41,317 replay steps, against 99,767 with short blocks.

## The token-scaled recipe starves small files of context

- Each block splits into 64 streams, and an online prediction sees only the earlier tokens of its own stream in that block.
- Scaling the block length by the token count gives a 10 MB file 83-token streams and a 1 MB file 7-token streams. The model has a 192-token context.
- Running the 100 MB recipe on 10 MB saves 1.7% to 3.1% and takes fewer replay steps.
- With short blocks 4×384 is 1.9% behind 12×768. With long blocks it is 0.4% behind at the same learning rates.
- With short blocks 256 streams cost 11% to 13%. With long blocks and a tuned online rate they cost 1.0%.
- 16 streams save 1.6% over 64 and take 2.6 times as long.
- 54,926-token blocks are the best of five block lengths from 13,731 to 219,704.

## What did not work

- `tensor / python_number` is a reciprocal multiply on CUDA and a division on the CPU. The two differ in the last bit at widths that are not powers of two. Every scalar division is now an explicit reciprocal multiply or a tensor division.
- The 1 MB bit-width sweep said 8/8/12 bits are free. On 10 MB they cost 7.9% to 11.6%.
- The float32 forward path for low bit widths is exact and no faster.
- More than 64 streams do not shorten a float run, because replay steps dominate and a float step costs the same at any batch size.
- The 100 MB deterministic time was first estimated at 13 hours from the 10 MB token rate. It took 17.9 hours, because 100 MB online steps use the full 192-token window.
- The 12×768 learning rate was not tuned on 100 MB.

## Still running

- The 100 MB decode on a B200.
