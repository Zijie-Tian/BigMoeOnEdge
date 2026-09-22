# MoE hit

Expert-cache sweep on one Jetson AGX (about 30991 MiB visible). Four pure
`Q4_0` models, the same protocol. `--cache-mb` runs from 0 through 28 GiB in
1 GiB steps. Non-expert weights are loaded once into anonymous memory.
Routed experts are read with `O_DIRECT`. A budget of 0 is the shared-slot
path: one layer-sized buffer, overwritten every layer, no LRU. A positive
budget is the per-`(layer, expert)` LRU.

Every run is traced (`--compute-trace` and `--io-trace`), so the tok/s below
include the trace barriers. Compare rows inside one table only. The prompt is
`The capital of Japan is`, 32 greedy tokens, two generations in one process
with `clear_kv=true`. The rows are the hot generation. `-t 4 -c 256
--ubatch 128 --io-threads 4`. The model file is dropped from the page cache
before the process. GPU off.

`cache_hit_pct` is the cumulative counter at the end of the hot generation.
Prefill and the cold generation are included, so a hot generation that reads
nothing can still show a hit rate below 100%. Expert read volume is that
generation's `read_mib` divided by 32. A budget below 1500 MiB is started
with `--force-cache`.

## Mixtral-8x7B

Base checkpoint, not the Instruct finetune. Pure `Q4_0`, file 25057.64 MiB.
32 layers, 8 experts, top-2. One expert is 94.5 MiB. One token touches
**6048 MiB** of routed experts (measured aligned read 6048.75 MiB with the
cache off). The 32-token trajectory actually retains **24003 MiB**, under the
24192 MiB routed bank. Non-expert anonymous set: 865 MiB. No shared experts.

Hot decode. Read time is `io_s_tok`, management is `mgmt_s_tok`, compute is
`compute_s_tok`.

| Cache | tok/s | Read | Management | Compute | Expert read | Cumulative hit | Resident |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.418 | 1.40 s | 0.00 s | 0.99 s | 6048.8 MiB/token | — | shared slot |
| 1 GiB | 0.153 | 4.42 s | 1.09 s | 1.01 s | 6048.8 | 0.0% | 945 MiB |
| 2 | 0.141 | 4.77 | 1.33 | 1.01 | 6048.8 | 0.0% | 1984 |
| 3 | 0.138 | 4.93 | 1.29 | 1.01 | 6048.8 | 0.0% | 3024 |
| 4 | 0.138 | 4.86 | 1.37 | 1.02 | 6048.8 | 0.0% | 4064 |
| 5 | 0.141 | 4.77 | 1.31 | 1.01 | 6048.8 | 0.0% | 5103 |
| 6 | 0.189 | 3.39 | 0.88 | 1.02 | 4061.1 | 30.0% | 6142 |
| 7 | 0.188 | 3.41 | 0.88 | 1.02 | 4043.3 | 30.4% | 7088 |
| 8 | 0.184 | 3.39 | 0.88 | 1.17 | 4022.7 | 30.7% | 8127 |
| 9 | 0.194 | 3.29 | 0.85 | 1.03 | 3963.6 | 31.7% | 9166 |
| 10 | 0.206 | 3.06 | 0.76 | 1.03 | 3606.2 | 37.3% | 10206 |
| 11 | 0.238 | 2.53 | 0.64 | 1.03 | 3065.7 | 45.6% | 11246 |
| 12 | 0.249 | 2.41 | 0.59 | 1.02 | 2853.1 | 49.0% | 12285 |
| 13 | 0.272 | 2.13 | 0.52 | 1.02 | 2543.0 | 54.0% | 13230 |
| 14 | 0.297 | 1.89 | 0.46 | 1.02 | 2232.8 | 58.8% | 14270 |
| 15 | 0.309 | 1.78 | 0.43 | 1.02 | 2117.7 | 60.8% | 15309 |
| 16 | 0.350 | 1.47 | 0.37 | 1.02 | 1701.2 | 67.4% | 16348 |
| 17 | 0.380 | 1.29 | 0.31 | 1.03 | 1521.0 | 70.6% | 17388 |
| 18 | 0.437 | 1.03 | 0.25 | 1.01 | 1225.7 | 75.2% | 18428 |
| 19 | 0.464 | 0.93 | 0.22 | 1.01 | 1072.1 | 77.9% | 19372 |
| 20 | 0.526 | 0.73 | 0.17 | 1.00 | 853.6 | 81.4% | 20412 |
| 21 | 0.610 | 0.52 | 0.12 | 1.00 | 605.5 | 85.4% | 21452 |
| 22 | 0.672 | 0.27 | 0.06 | 1.16 | 307.2 | 90.1% | 22491 |
| 23 | 0.934 | 0.07 | 0.02 | 0.99 | 76.8 | 93.3% | 23530 |
| 24 | 1.006 | 0 | 0.00 | 0.99 | 0 | 94.3% | 24003 |
| 25 | 1.010 | 0 | 0.00 | 0.99 | 0 | 94.3% | 24003 |
| 26 | 1.005 | 0 | 0.00 | 1.00 | 0 | 94.3% | 24003 |
| 27 | 1.023 | 0 | 0.00 | 0.98 | 0 | 94.3% | 24003 |
| 28 | 1.013 | 0 | 0.00 | 0.99 | 0 | 94.3% | 24003 |

Below 6 GiB the budget is smaller than one token, the hit rate stays 0, and
every token still reads 6048.8 MiB. Those rows are slower than cache 0: the
read phase grows from 1.40 s to about 4.4–4.9 s, and eviction adds about
1.1–1.4 s, while compute stays near 1.0 s. From 6 GiB the read volume falls.
At 24 GiB the trajectory is fully resident, the hot generation reads nothing,
and tok/s sits near 1.01. Larger budgets do not raise the resident set past
24003 MiB.

## DeepSeek-V2-Lite

Pure `Q4_0`, file 8,851,045,376 bytes. 26 routed layers, 64 experts, top-6,
plus one leading dense layer. One token touches **724 MiB** of routed experts
(measured aligned read 725.8 MiB with the cache off). The 32-token trajectory
retains **5792 MiB**, under the 7722 MiB routed bank. Non-expert anonymous
set, including the shared experts: about 715 MiB.

Same host, same prompt, same trace settings, same hot-generation columns.

| Cache | tok/s | Read | Management | Compute | Expert read | Cumulative hit | Resident |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.395 | 0.17 s | 0.00 s | 0.25 s | 725.8 MiB/token | — | shared slot |
| 1 GiB | 1.330 | 0.36 s | 0.12 s | 0.28 s | 490.0 | 28.9% | 1021 MiB |
| 2 | 1.564 | 0.28 | 0.08 | 0.28 | 358.5 | 45.6% | 2047 |
| 3 | 1.821 | 0.22 | 0.06 | 0.27 | 271.0 | 57.8% | 3067 |
| 4 | 2.450 | 0.11 | 0.03 | 0.27 | 131.4 | 75.1% | 4093 |
| 5 | 2.950 | 0.04 | 0.01 | 0.28 | 47.5 | 84.4% | 5119 |
| 6 | 3.889 | 0 | 0.00 | 0.26 | 0 | 88.9% | 5792 |
| 7 | 3.880 | 0 | 0.00 | 0.26 | 0 | 88.9% | 5792 |
| 8 | 3.874 | 0 | 0.00 | 0.26 | 0 | 88.9% | 5792 |
| 9 | 3.880 | 0 | 0.00 | 0.26 | 0 | 88.9% | 5792 |
| 10 | 3.699 | 0 | 0.00 | 0.27 | 0 | 88.9% | 5792 |
| 11 | 3.914 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 12 | 3.750 | 0 | 0.00 | 0.27 | 0 | 88.9% | 5792 |
| 13 | 3.936 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 14 | 3.927 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 15 | 3.930 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 16 | 3.925 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 17 | 3.747 | 0 | 0.00 | 0.27 | 0 | 88.9% | 5792 |
| 18 | 3.922 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 19 | 3.747 | 0 | 0.00 | 0.27 | 0 | 88.9% | 5792 |
| 20 | 3.922 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 21 | 3.913 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 22 | 3.937 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 23 | 3.925 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 24 | 3.700 | 0 | 0.00 | 0.27 | 0 | 88.9% | 5792 |
| 25 | 3.921 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 26 | 3.689 | 0 | 0.00 | 0.27 | 0 | 88.9% | 5792 |
| 27 | 3.926 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |
| 28 | 3.915 | 0 | 0.00 | 0.25 | 0 | 88.9% | 5792 |

One token is only 724 MiB, so 1 GiB already records a 28.9% cumulative hit
and cuts the hot read from 725.8 to 490.0 MiB/token. At 1–3 GiB that saving
does not beat the extra read and management time, and tok/s stays below
cache 0. From 4 GiB the saved reads win. At 6 GiB the 5792 MiB trajectory is
resident, the hot generation reads nothing, and tok/s sits near 3.9. Larger
budgets do not change the resident set.

## Qwen1.5-MoE-A2.7B

Pure `Q4_0`, quant size 7690.23 MiB (4.51 BPW). 24 layers, 60 experts, top-4.
One token's measured aligned read with the cache off is **237.3 MiB**. The
32-token trajectory retains **4720 MiB**. Non-expert anonymous set, including
the shared expert: **1007 MiB**.

Same host and protocol as the tables above.

| Cache | tok/s | Read | Management | Compute | Expert read | Cumulative hit | Resident |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.822 | 0.11 s | 0.00 s | 0.25 s | 237.3 MiB/token | — | shared slot |
| 1 GiB | 1.676 | 0.27 s | 0.08 s | 0.25 s | 187.3 | 17.6% | 1021 MiB |
| 2 | 1.946 | 0.21 | 0.05 | 0.25 | 138.4 | 36.1% | 2047 |
| 3 | 2.183 | 0.17 | 0.04 | 0.25 | 108.8 | 46.9% | 3067 |
| 4 | 2.471 | 0.13 | 0.03 | 0.25 | 76.3 | 56.6% | 4093 |
| 5 | 3.948 | 0 | 0.00 | 0.25 | 0 | 74.4% | 4720 |
| 6 | 4.117 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 7 | 4.095 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 8 | 4.121 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 9 | 4.111 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 10 | 4.123 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 11 | 4.121 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 12 | 4.043 | 0 | 0.00 | 0.25 | 0 | 74.4% | 4720 |
| 13 | 4.108 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 14 | 4.135 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 15 | 4.122 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 16 | 4.119 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 17 | 4.117 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 18 | 4.110 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 19 | 4.072 | 0 | 0.00 | 0.25 | 0 | 74.4% | 4720 |
| 20 | 4.118 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 21 | 4.102 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 22 | 4.134 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 23 | 4.122 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 24 | 4.128 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 25 | 4.120 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 26 | 4.133 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 27 | 4.123 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |
| 28 | 4.125 | 0 | 0.00 | 0.24 | 0 | 74.4% | 4720 |

One token is 237 MiB, so 1 GiB already hits. At 1–4 GiB the LRU overhead still
leaves tok/s below cache 0. At 5 GiB the 4720 MiB trajectory is resident, the
hot generation reads nothing, and tok/s sits near 4.1. Larger budgets do not
move the resident set.

## Qwen2-57B-A14B

Base checkpoint, not the Instruct finetune. Pure `Q4_0`, quant size 30818.88 MiB
(4.50 BPW). 28 layers, 64 experts, top-8. One token's measured aligned read
with the cache off is **3310.1 MiB**. The 32-token trajectory retains
**21528 MiB**, under the 26460 MiB routed bank. Non-expert anonymous set,
including the shared expert: **4358 MiB**.

Same host and protocol. Every budget through 28 GiB completed.

| Cache | tok/s | Read | Management | Compute | Expert read | Cumulative hit | Resident |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.520 | 0.72 s | 0.00 s | 1.20 s | 3310.1 MiB/token | — | shared slot |
| 1 GiB | 0.256 | 2.04 s | 0.72 s | 1.14 s | 3310.1 | 0.0% | 1019 MiB |
| 2 | 0.247 | 2.05 | 0.72 | 1.28 | 3310.1 | 0.0% | 2038 |
| 3 | 0.258 | 2.04 | 0.67 | 1.16 | 3310.1 | 0.0% | 3071 |
| 4 | 0.323 | 1.48 | 0.49 | 1.13 | 2267.9 | 28.7% | 4090 |
| 5 | 0.317 | 1.47 | 0.48 | 1.20 | 2238.8 | 29.6% | 5109 |
| 6 | 0.344 | 1.30 | 0.41 | 1.20 | 1939.1 | 37.9% | 6142 |
| 7 | 0.362 | 1.27 | 0.39 | 1.10 | 1874.0 | 40.0% | 7161 |
| 8 | 0.373 | 1.19 | 0.36 | 1.13 | 1721.6 | 44.3% | 8180 |
| 9 | 0.369 | 1.10 | 0.34 | 1.27 | 1624.6 | 47.3% | 9214 |
| 10 | 0.395 | 1.03 | 0.31 | 1.19 | 1493.0 | 51.2% | 10233 |
| 11 | 0.410 | 0.96 | 0.29 | 1.19 | 1385.8 | 54.4% | 11251 |
| 12 | 0.468 | 0.80 | 0.23 | 1.11 | 1167.9 | 60.7% | 12285 |
| 13 | 0.457 | 0.72 | 0.20 | 1.27 | 1014.1 | 65.0% | 13304 |
| 14 | 0.513 | 0.61 | 0.16 | 1.17 | 892.6 | 68.4% | 14323 |
| 15 | 0.557 | 0.54 | 0.14 | 1.12 | 727.8 | 73.1% | 15356 |
| 16 | 0.548 | 0.45 | 0.11 | 1.26 | 627.1 | 75.9% | 16375 |
| 17 | 0.657 | 0.36 | 0.09 | 1.08 | 489.0 | 79.6% | 17394 |
| 18 | 0.626 | 0.29 | 0.07 | 1.24 | 389.3 | 82.4% | 18428 |
| 19 | 0.694 | 0.21 | 0.05 | 1.19 | 254.9 | 85.4% | 19446 |
| 20 | 0.814 | 0.10 | 0.03 | 1.10 | 119.1 | 88.6% | 20480 |
| 21 | 0.761 | 0.04 | 0.01 | 1.26 | 48.0 | 89.9% | 21499 |
| 22 | 0.798 | 0 | 0.00 | 1.25 | 0 | 90.8% | 21528 |
| 23 | 0.853 | 0 | 0.00 | 1.17 | 0 | 90.8% | 21528 |
| 24 | 0.891 | 0 | 0.00 | 1.12 | 0 | 90.8% | 21528 |
| 25 | 0.802 | 0 | 0.00 | 1.25 | 0 | 90.8% | 21528 |
| 26 | 0.849 | 0 | 0.00 | 1.18 | 0 | 90.8% | 21528 |
| 27 | 0.846 | 0 | 0.00 | 1.18 | 0 | 90.8% | 21528 |
| 28 | 0.825 | 0 | 0.00 | 1.21 | 0 | 90.8% | 21528 |

One token is 3310 MiB, so 1–3 GiB stay at a 0% hit and the full 3310.1 MiB
read, and they are slower than cache 0. Hits begin at 4 GiB. The read volume
then falls with the budget. At 22 GiB the 21528 MiB trajectory is resident,
the hot generation reads nothing, and tok/s sits near 0.8–0.9. Compute stays
near 1.1–1.3 s at every budget. Larger caches do not raise the resident set
past 21528 MiB.

