# Model size scaling on enwik8

## Setup of the 10 MB sweep

- Data: the first 10,000,000 bytes of enwik8, tokenized to 3,065,672 tokens with a 4,096-word dictionary.
- Mode: float bf16, random initial weights, 64 streams, one H100.
- Recipe: the upstream schedule with block length, replay window and replay schedule scaled by the token ratio 3,065,672 / 285,202,830.
- Shapes: head dimension 128 and inner width 2.75 × model width at every size.
- Runs: six sizes, five learning rates each with seed 1111, and a second seed at the best rate.
- Bytes are the modeled size of the token stream. The dictionary adds 28,885 bytes to every row.

## A 10.4M-parameter model is 2.1% behind the 162.6M-parameter original

The table shows the best learning rate for each size, run with two seeds.

| Size | Parameters | Learning rate | Seed 1111 | Seed 2222 | Mean bytes | Behind best | Minutes |
|---|---|---|---|---|---|---|---|
| 2×256 | 3.7M | 1.2e-3 | 2,184,623 | 2,184,196 | 2,184,410 | +5.9% | 29 |
| 4×384 | 10.4M | 1e-3 | 2,103,812 | 2,107,549 | 2,105,680 | +2.1% | 36 |
| 6×512 | 23.5M | 1.2e-3 | 2,086,817 | 2,092,222 | 2,089,520 | +1.3% | 54 |
| 8×640 | 45.4M | 5e-4 | 2,086,454 | 2,077,542 | 2,081,998 | +0.9% | 72 |
| 12×768 | 93.0M | 8e-4 | 2,065,528 | 2,070,030 | 2,067,779 | +0.2% | 116 |
| 12×1024 | 162.6M | 2.5e-4 | 2,055,182 | 2,071,092 | 2,063,137 | best | 165 |

- 12×1024 is the upstream architecture.
- 4×384 is the smallest size within 3% of the best mean.
- The largest step is from 2×256 to 4×384: 3.6% fewer bytes for 7 more minutes.
- From 4×384 to 12×1024 the model has 16 times the parameters and takes 4.6 times the wall time for 2.0% fewer bytes.
- Two seeds of one size differ by 0.02% to 0.77%.
- Minutes are from the seed 1111 run.

The plots show the seed 1111 runs.

![Compressed size by parameter count](scaling-enwik8-10MB-parameters.png)

![Compressed size by wall time](scaling-enwik8-10MB-minutes.png)

## Standard compressors on the same 10 MB

| Compressor | Bytes |
|---|---|
| gzip -9 | 3,688,470 |
| bzip2 -9 | 2,916,026 |
| zstd -19 | 2,795,185 |
| xz -9e | 2,720,264 |
| 4×384 plus dictionary | 2,132,697 |

## The best learning rate is near 1e-3 for the small sizes

- The best rate is bracketed at every size: a higher and a lower rate both give more bytes.
- The best rate is 1e-3 to 1.2e-3 for 2×256, 4×384 and 6×512.
- The upstream online rate of 4e-5 is the worst of the five rates tried at 12×1024.
- Doubling the best rate costs between 0.02% (4×384) and 1.1% (12×768).

## With short blocks, 256 streams cost 11% to 13% more bytes than 64

| Size | Streams | Learning rate | Bytes | Against 64 streams | Minutes |
|---|---|---|---|---|---|
| 4×384 | 64 | 1e-3 | 2,103,812 | | 36 |
| 4×384 | 256 | 1e-3 | 2,380,157 | +13.1% | 43 |
| 4×384 | 256 | 2e-3 | 2,375,690 | +12.9% | 28 |
| 6×512 | 64 | 1.2e-3 | 2,086,817 | | 54 |
| 6×512 | 256 | 1.2e-3 | 2,333,933 | +11.8% | 41 |
| 6×512 | 256 | 2.4e-3 | 2,342,999 | +12.3% | 35 |

- With 256 streams the model takes a quarter as many online steps.
- Wall time falls by less than a quarter, because replay steps do not depend on the stream count.
- With 256 streams each stream covers 20 tokens of a block, so an online prediction sees at most 19 tokens of context.

## Limits of these numbers

- The learning-rate sweep is one seed per cell, and seed noise is up to 0.8%.
- Wall time varies between runs of the same size: 2×256 took between 16 and 30 minutes.
- The file is 10 MB. A larger file gives a larger model more data to learn from.
- The scaled recipe caps the online context at 82 tokens. The next section measures what that costs.

## Longer blocks save 1.7% to 3.1% on 10 MB and shrink the size gap

Each block splits into 64 streams, and an online prediction sees only the earlier tokens of its own stream in that block.

- The scaled 10 MB recipe has 5,375-token blocks, so a stream is 83 tokens and the context averages 41 tokens.
- The 100 MB recipe has 54,926-token blocks, so a stream is 858 tokens and the context averages 170 tokens.
- The 1 MB recipe has 470-token blocks, so the context is at most 6 tokens.

The table runs the 100 MB recipe on the 10 MB slice, one seed, at the learning rates of the main table.

| Size | Scaled 10 MB recipe | 100 MB recipe | Change | Minutes, scaled | Minutes, 100 MB recipe |
|---|---|---|---|---|---|
| 4×384 | 2,103,812 | 2,038,742 | −3.1% | 36 | 21 |
| 6×512 | 2,086,817 | 2,031,852 | −2.6% | 54 | 41 |
| 12×768 | 2,065,528 | 2,031,283 | −1.7% | 116 | 100 |

- The 100 MB recipe also takes 41,317 replay steps where the scaled recipe takes 99,767, so the runs are shorter.
- 4×384 is 0.4% behind 12×768 with the 100 MB recipe, against 1.9% with the scaled recipe.
- The learning rates were tuned for the scaled recipe.
- The 10 MB sweep above measures model size under short contexts. The 100 MB runs do not have this limit.

## With long blocks, 256 streams cost 1.0% and 16 streams save 1.6%

4×384 on 10 MB with the 100 MB recipe, seed 1111. The replay schedule is the same in every row and only the online learning rate varies.

| Streams | Tokens per stream in a block | Online steps | Online rates tried | Best rate | Bytes | Against 64 streams | Minutes |
|---|---|---|---|---|---|---|---|
| 16 | 3,432 | 191,605 | 5e-4, 1e-3, 2e-3 | 1e-3 | 2,005,343 | −1.64% | 54 |
| 32 | 1,716 | 95,803 | 5e-4, 1e-3, 2e-3 | 1e-3 | 2,011,381 | −1.34% | 33 |
| 64 | 858 | 47,902 | 1e-3 | 1e-3 | 2,038,742 | | 21 |
| 128 | 429 | 23,951 | 1e-3, 2e-3, 4e-3 | 4e-3 | 2,032,322 | −0.31% | 20 |
| 256 | 214 | 11,976 | 1e-3, 2e-3, 4e-3, 8e-3 | 4e-3 | 2,059,578 | +1.02% | 23 |

- The best online rate rises with the stream count: 1e-3 at 16 and 32 streams, 4e-3 at 128 and 256.
- 256 streams at the 64-stream rate of 1e-3 give 2,144,115 bytes, 5.2% more than 64 streams.
- Float wall time does not fall past 64 streams. Every row takes about 41,300 replay steps, and a float step costs the same at any batch size on an H100.
- 16 streams take 2.6 times as long as 64 streams for 1.6% fewer bytes.
- A second seed at 64 streams gives 2,034,435, so differences under 0.3% are noise.
- The 128-stream best is the highest rate tried.
- The 11% to 13% loss of the short-block table came from the 19-token context, which long blocks remove.

## 54,926-token blocks are the best of five block lengths

4×384 on 10 MB, 64 streams, learning rate 1e-3, seed 1111. Only the block length changes.

| Block length | Tokens per stream in a block | Replay steps | Bytes | Against 54,926 | Minutes |
|---|---|---|---|---|---|
| 13,731 | 214 | 167,132 | 2,076,864 | +1.87% | 55 |
| 27,463 | 429 | 83,388 | 2,054,818 | +0.79% | 35 |
| 54,926 | 858 | 41,317 | 2,038,742 | | 21 |
| 109,852 | 1,716 | 20,282 | 2,058,500 | +0.97% | 18 |
| 219,704 | 3,432 | 9,765 | 2,081,062 | +2.08% | 21 |

- A replay runs after every block, so half the block length means twice the replay steps.
- Four times the replay steps cost 1.87% more bytes and 34 more minutes.
- The replay learning-rate schedule counts replay steps and was not rescaled, so a run with more replay steps also ends at a lower replay rate. The table measures both changes together.

## On the full 100 MB, 6×512 is 3.0% ahead of 4×384

- Data: all 100,000,000 bytes of enwik8, tokenized to 31,330,083 tokens. The dictionary adds 29,081 bytes.
- Mode: float bf16, 64 streams, seed 1111, one run per size.
- Recipe: the upstream schedule scaled by the token ratio 31,330,083 / 285,202,830.
- Learning rates are the best ones from the 10 MB sweep.

| Size | Parameters | Learning rate | Bytes | Bits per byte | Behind best | Minutes | GPU |
|---|---|---|---|---|---|---|---|
| 4×384 | 10.4M | 1e-3 | 16,362,011 | 1.3090 | +3.1% | 329 | H100 |
| 6×512 | 23.5M | 1.2e-3 | 15,867,295 | 1.2694 | best | 543 | H100 |
| 12×768 | 93.0M | 8e-4 | 15,882,829 | 1.2706 | +0.1% | 909 | B200 |

- 6×512 saves 494,716 bytes over 4×384 for 214 more minutes.
- 12×768 has four times the parameters of 6×512 and ends 0.1% behind it.
- The 12×768 learning rate was tuned on 10 MB. A rate tuned on 100 MB may change its rank.

## The gap between 4×384 and 6×512 grows with the amount of data

The table shows the bytes written so far at five points of the 100 MB runs, read from the run logs.

| Tokens coded | 4×384 | 6×512 | 12×768 | 4×384 behind 6×512 |
|---|---|---|---|---|
| 1.10M | 787,117 | 784,047 | 780,390 | +0.4% |
| 3.08M | 1,944,583 | 1,932,686 | 1,932,404 | +0.6% |
| 9.99M | 5,622,356 | 5,543,705 | 5,589,838 | +1.4% |
| 19.99M | 10,685,956 | 10,435,364 | 10,484,487 | +2.4% |
| 31.33M | 16,362,011 | 15,867,295 | 15,882,829 | +3.1% |

- 4×384 loses ground at every point.
- 12×768 leads for the first 3M tokens, is 0.8% behind 6×512 at 10M tokens, and closes to 0.1% by the end.

## Standard compressors on the full 100 MB

| Compressor | Bytes |
|---|---|
| gzip -9 | 36,475,811 |
| bzip2 -9 | 29,008,758 |
| zstd -19 | 26,944,227 |
| xz -9e | 24,831,656 |
| 4×384 plus dictionary | 16,391,092 |
| 6×512 plus dictionary | 15,896,376 |

## All 30 runs of the learning-rate sweep

| Size | Parameters | Learning rate | Bytes | Bits per byte | Behind best | Minutes |
|---|---|---|---|---|---|---|
| 2×256 | 3.7M | 0.0002 | 2,205,856 | 1.7647 | +7.3% | 23 |
| 2×256 | 3.7M | 0.0005 | 2,185,929 | 1.7487 | +6.4% | 30 |
| 2×256 | 3.7M | 0.0012 | **2,184,623** | 1.7477 | +6.3% | 29 |
| 2×256 | 3.7M | 0.0024 | 2,187,793 | 1.7502 | +6.5% | 17 |
| 2×256 | 3.7M | 0.0048 | 2,195,236 | 1.7562 | +6.8% | 16 |
| 4×384 | 10.4M | 0.00015 | 2,114,237 | 1.6914 | +2.9% | 40 |
| 4×384 | 10.4M | 0.0004 | 2,113,158 | 1.6905 | +2.8% | 39 |
| 4×384 | 10.4M | 0.001 | **2,103,812** | 1.6830 | +2.4% | 36 |
| 4×384 | 10.4M | 0.002 | 2,104,265 | 1.6834 | +2.4% | 37 |
| 4×384 | 10.4M | 0.004 | 2,116,510 | 1.6932 | +3.0% | 29 |
| 6×512 | 23.5M | 0.0001 | 2,100,138 | 1.6801 | +2.2% | 77 |
| 6×512 | 23.5M | 0.00025 | 2,101,263 | 1.6810 | +2.2% | 79 |
| 6×512 | 23.5M | 0.0006 | 2,089,624 | 1.6717 | +1.7% | 45 |
| 6×512 | 23.5M | 0.0012 | **2,086,817** | 1.6695 | +1.5% | 54 |
| 6×512 | 23.5M | 0.0024 | 2,103,643 | 1.6829 | +2.4% | 51 |
| 8×640 | 45.4M | 8e-05 | 2,094,652 | 1.6757 | +1.9% | 75 |
| 8×640 | 45.4M | 0.0002 | 2,098,320 | 1.6787 | +2.1% | 66 |
| 8×640 | 45.4M | 0.0005 | **2,086,454** | 1.6692 | +1.5% | 72 |
| 8×640 | 45.4M | 0.001 | 2,103,663 | 1.6829 | +2.4% | 104 |
| 8×640 | 45.4M | 0.002 | 2,104,753 | 1.6838 | +2.4% | 109 |
| 12×768 | 93.0M | 6e-05 | 2,097,687 | 1.6781 | +2.1% | 120 |
| 12×768 | 93.0M | 0.00015 | 2,078,868 | 1.6631 | +1.2% | 119 |
| 12×768 | 93.0M | 0.0004 | 2,069,021 | 1.6552 | +0.7% | 119 |
| 12×768 | 93.0M | 0.0008 | **2,065,528** | 1.6524 | +0.5% | 116 |
| 12×768 | 93.0M | 0.0016 | 2,087,811 | 1.6702 | +1.6% | 114 |
| 12×1024 | 162.6M | 4e-05 | 2,084,760 | 1.6678 | +1.4% | 165 |
| 12×1024 | 162.6M | 0.0001 | 2,072,431 | 1.6579 | +0.8% | 151 |
| 12×1024 | 162.6M | 0.00025 | **2,055,182** | 1.6441 | best | 165 |
| 12×1024 | 162.6M | 0.0005 | 2,065,880 | 1.6527 | +0.5% | 148 |
| 12×1024 | 162.6M | 0.001 | 2,059,870 | 1.6479 | +0.2% | 166 |

Regenerate the plots and this table with `uv run python scripts/plot_scaling.py results/ledger.jsonl enwik8-10MB results`.
