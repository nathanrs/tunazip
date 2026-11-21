# Deterministic mode: devices, bytes and speed

## Setup

- Model: 4×384, 10.4M parameters, 64 streams, random initial weights.
- Deterministic mode rounds every matmul operand to 16 significant bits and runs the matmul in float64, where the result is exact.
- Float mode is bf16 autocast, the upstream setting.
- Bytes are the modeled size of the token stream unless a row says payload.
- Encodes ran on one H100.
- Two recipes appear. The short-block recipe has 5,375-token blocks on 10 MB, which caps the online context at 82 tokens. The long-block recipe has 54,926-token blocks, which gives the full 192-token context. `scaling.md` measures the difference.
- The round trips and the first bit-width tables use the short-block recipe.

## A file encoded on an H100 decodes byte-exactly on five other GPU types

Each decode reproduced the token stream and restored the original file bytes through the word preprocessor.

| Slice | Payload bytes | Decoder | GPU generation | Decode minutes | Restores the original bytes |
|---|---|---|---|---|---|
| 1 MB | 275,620 | B200 | Blackwell | 6.8 | yes |
| 1 MB | 275,620 | A100 | Ampere | 11.3 | yes |
| 1 MB | 275,620 | L40S | Ada | 48.0 | yes |
| 1 MB | 275,620 | A10 | Ampere | 136.0 | yes |
| 1 MB | 275,620 | T4 | Turing | 253.1 | yes |
| 10 MB | 2,101,694 | B200 | Blackwell | 71.7 | yes |
| 10 MB | 2,101,694 | A100 | Ampere | 135.9 | yes |

- The H100 encodes took 6.6 minutes for 1 MB and 76.7 minutes for 10 MB.
- Decode time follows float64 throughput. The L40S, A10 and T4 are 7 to 39 times slower than the H100 on the 1 MB file.

Two cheaper checks run on every device before a round trip:

- 31 operation hashes are identical on H100, B200, A100, L40S, A10, T4, a Linux CPU and a macOS CPU.
- A 16,000-token training run ends with the same parameter hash on the six GPU types and the CPU.

## Determinism costs no measurable bytes at 10 MB

| Recipe | Mode | Seed 1111 | Seed 2222 | Mean bytes |
|---|---|---|---|---|
| Short blocks | Float bf16 | 2,103,812 | 2,107,549 | 2,105,680 |
| Short blocks | Deterministic, 16 bits | 2,101,163 | 2,100,320 | 2,100,742 |
| Long blocks | Float bf16 | 2,038,742 | 2,034,435 | 2,036,589 |
| Long blocks | Deterministic, 16 bits | 2,037,784 | 2,042,855 | 2,040,320 |

- With short blocks the deterministic mean is 0.23% smaller than the float mean.
- With long blocks the deterministic mean is 0.18% larger than the float mean.
- Two seeds of one mode differ by 0.04% to 0.25%, so the two modes are within seed noise of each other.
- On the 1 MB slice the deterministic mean is 275,240 bytes against 275,180 for float, a difference of 0.02%.

## Deterministic mode takes 3.0 to 3.4 times as long as float with full contexts

| Slice | Recipe | Float minutes | Deterministic minutes | Ratio |
|---|---|---|---|---|
| 1 MB | Short blocks | 3.3 to 3.7 | 6.5 | 1.8 to 2.0 |
| 10 MB | Short blocks | 35.7 and 52.9 | 76.7 and 77.1 | 1.5 to 2.2 |
| 10 MB | Long blocks | 20.7 and 23.3 | 70.6 and 70.3 | 3.0 to 3.4 |

- With long blocks most online steps see 192 tokens of context, and that step is where deterministic mode is slowest.
- With short blocks the online steps see 41 tokens on average, so the ratio is lower.
- Float wall time varies with the host CPU. Deterministic wall time varies by under 1%.
- Per step on an H100, a replay step takes 37.1 ms deterministic and 15.7 ms float.
- An online step over a 192-token window takes 62.8 ms deterministic and 16.9 ms float.
- At 12×768 the deterministic replay step takes 218.0 ms against 49.2 ms for float, a ratio of 4.4.

The deterministic step replays as a CUDA graph of fused kernels. Run eagerly it took 140 ms per step at 4×384.

## Gradients need 16 bits at 10 MB

Each column is the number of significant bits kept for one matmul operand role. Runs are deterministic, 4×384, on an H100.

| Activation bits | Weight bits | Gradient bits | Seed 1111 | Seed 2222 | Against 16/16/16 |
|---|---|---|---|---|---|
| 16 | 16 | 16 | 2,101,163 | 2,100,320 | baseline |
| 16 | 8 | 16 | 2,097,315 | | −0.18% |
| 12 | 12 | 16 | 2,098,903 | | −0.11% |
| 8 | 16 | 16 | 2,107,589 | | +0.31% |
| 8 | 8 | 16 | 2,110,768 | | +0.46% |
| 16 | 16 | 12 | 2,163,193 | | +2.95% |
| 8 | 8 | 12 | 2,267,327 | 2,343,655 | +7.9% and +11.6% |
| 6 | 6 | 12 | 2,387,470 | 2,520,514 | +13.6% and +20.0% |

- 12-bit gradients alone cost 2.95%.
- 8-bit activations and weights cost at most 0.46% while gradients keep 16 bits.
- The two losses compound: 8-bit activations and weights with 12-bit gradients cost 7.9% to 11.6%.
- The low-bit runs are unstable. Two seeds of 8/8/12 differ by 3.4%, against 0.04% at 16 bits.
- Rows with one seed carry seed noise of a few tenths of a percent.

## With full contexts 12-bit gradients cost 5.2%

The same roles on 10 MB with the long-block recipe, seed 1111.

| Activation bits | Weight bits | Gradient bits | Bytes | Against 16/16/16 |
|---|---|---|---|---|
| 16 | 16 | 16 | 2,037,784 | baseline |
| 8 | 8 | 16 | 2,029,517 | −0.41% |
| 12 | 12 | 16 | 2,047,024 | +0.45% |
| 16 | 16 | 20 | 2,043,323 | +0.27% |
| 16 | 16 | 12 | 2,144,017 | +5.21% |

- 12-bit gradients cost 5.21% here, against 2.95% with short blocks.
- 20-bit gradients do not beat 16-bit gradients.
- 8-bit and 12-bit activations and weights stay within 0.5% of the baseline, on either side of it.
- A second seed of 16/16/16 gives 2,042,855, which is 0.25% above the baseline.

## The 1 MB slice hides the gradient requirement

The same sweep on the first 1 MB, two seeds per row, against the float mean of 275,180 bytes.

| Activation bits | Weight bits | Gradient bits | Mean bytes | Against float |
|---|---|---|---|---|
| 16 | 16 | 16 | 275,240 | +0.02% |
| 16 | 8 | 16 | 274,785 | −0.14% |
| 8 | 16 | 16 | 274,582 | −0.22% |
| 8 | 8 | 16 | 274,606 | −0.21% |
| 6 | 6 | 16 | 275,083 | −0.04% |
| 4 | 4 | 16 | 279,086 | +1.42% |
| 12 | 12 | 12 | 275,196 | +0.01% |
| 8 | 8 | 12 | 275,061 | −0.04% |
| 8 | 8 | 10 | 283,610 | +3.06% |
| 16 | 16 | 8 | 292,851 | +6.42% |
| 12 | 16 | 8 | 293,176 | +6.54% |
| 8 | 16 | 8 | 292,966 | +6.46% |
| 8 | 8 | 8 | 293,309 | +6.59% |

- 8/8/12 matches float on 1 MB and loses 7.9% to 11.6% on 10 MB.
- A bit width has to be checked at the size of the file it will compress.
- Lower bit widths buy no speed on an H100. A float32 forward matmul at 6 bits moved the replay step from 37.1 ms to 36.5 ms.
