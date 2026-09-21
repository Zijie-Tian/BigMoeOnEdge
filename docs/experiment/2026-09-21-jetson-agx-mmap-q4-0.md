# Jetson AGX mmap vs streaming, DeepSeek-V2-Lite Q4_0

This is a **host-side experiment record** from 2026-09-21. It is not an Android
device table and it does not replace [benchmarks.md](../benchmarks.md) or
[cpu-profiling-findings.md](../cpu-profiling-findings.md).

The question that drove the campaign: under **ordinary llama.cpp mmap** (no
`--moe-stream`), does extra RAM keep more of a MoE in the page cache and
speed up hot decode? A first arm used `--moe-stream` with Linux `O_DIRECT`
and `--cache-mb 0`. That arm does **not** answer the mmap question: expert
reads bypass the page cache and the engine does not retain experts across
tokens. The mmap arm does.

Primary result: **hot mmap decode** (same process, second generation,
`clear_kv=true`). Cold mmap and the O_DIRECT streaming arm are recorded
because they were run; they are not the headline.

A traced run is not a throughput benchmark. `--compute-trace` isolates every
graph node. Read proportions and the mmap majflt signal. Untraced tok/s is
not claimed.

## Snapshot

| Item | Value |
|---|---|
| Date | 2026-09-21 |
| Engine | 0.23.0 |
| Code | `tzj/bitmoe` at `416b7a9` |
| llama.cpp pin | `0e8c83e512b99ccf83e50798b86f0b0ec40a7b0a` |
| Model | DeepSeek-V2-Lite, architecture `deepseek2` |
| Quant used for the tables below | **pure `Q4_0`** (`llama-quantize --pure Q4_0` from F16) |
| Prompt | `The capital of Japan is` |
| Sampling | greedy (`temp=0`) |
| `-n` / `-c` / `-t` | 32 / 256 / 4 |
| GPU | off (`GGML_CUDA=OFF`, `n_gpu_layers=0`, no CUDA libs) |

### Hosts

Three Jetson AGX boards, same SoC class: 8 Carmel cores, ARMv8.2-A, FP16
vector arithmetic, **no DOTPROD**. L4T R35.6.x, NVMe root. They differ by
how much RAM Linux can see. The 8 GiB and 4 GiB caps are kernel cmdline
`mem=`, not cgroup `memory.max` (cgroup v1 limit was unlimited).

| Label | Visible RAM (`MemTotal`) | Swap | `mem=` |
|---|---|---|---|
| Host A | ~30 GiB | ~15 GiB | none |
| Host B | 7.6 GiB | 3.8 GiB zram | `8192M` |
| Host C | 3.6 GiB | 1.8 GiB zram | `4096M` |

The soldered DRAM on an AGX is larger than the 4/8 GiB caps. Do not quote
`MemTotal` as the package size.

### Build (each board, on device)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16
cmake --build build -j 4   # -j 2 on the 4 GiB host
```

CMake reported ARM, `-march=armv8.2-a+fp16`, `HAVE_DOTPROD` failed,
`HAVE_FP16_VECTOR_ARITHMETIC` succeeded. `bmoe-cli --version` is 0.23.0.
`ldd` has no CUDA libraries. Every layer printed `assigned to device CPU`.

## Model conversion

Hugging Face DeepSeek-V2-Lite → pinned `convert_hf_to_gguf.py --outtype f16`
(CPU) → native `llama-quantize`.

| File | Bytes | bpw | `general.file_type` |
|---|---:|---:|---|
| F16 | 31,424,035,840 | 16.00 | — |
| Q4_K_M (default mix; **not** the mmap tables) | 10,367,958,016 | 5.28 | 15 |
| **Q4_0 `--pure`** | **8,851,045,376** | **4.51** | **2** |

Architecture facts (unchanged by quant): 27 layers, 1 leading dense, 26 MoE,
64 routed experts, top-6, 2 shared experts. Shared experts and MLA stay on
the dense side of the seam.

### Why Q4_0, not Q4_K_M

`Q4_K_M` is llama.cpp's **default mixed k-quant**. The engine can run it
(per-tensor types, kernel dispatch). It is not uniform 4-bit. Inspecting
that file: **zero `Q4_0` tensors**. Routed experts were `Q4_K` (gate/up,
52 tensors, 5148 MiB) plus `Q5_0` / `Q8_0` down projections (14 + 12
tensors). Shared experts were `Q4_K` + `Q6_K`. Average 5.28 bpw.

`--pure Q4_0` from the same F16 makes **all 78** `ffn_{gate,up,down}_exps`
tensors `Q4_0` (7,722 MiB). The only F32 left are 108 norm / router
(`ffn_gate_inp`) tensors, not mixed expert weights.

| Quantity (Q4_0) | Size |
|---|---|
| File | 8,851,045,376 B (8.45 GiB) |
| Routed expert bank | 7,722 MiB |
| Non-routed (dense / MLA / shexp / layer-0 FFN, Q4_0+F32) | ~735 MiB of weights; anon dense conversion measured **715 MiB** |
| One `(layer, expert)` (uniform) | ≈ 4.64 MiB |
| One decode token, 26 × top-6 | ≈ 724 MiB (measured stream reads **725.77 MiB/token**) |

Host x86 CPU smoke, same Q4_0 file, 32 tokens, greedy: streamed
(`--moe-stream --cache-mb 0 --dense-weights anon`) generated text **byte-equal**
to mmap-resident. Not a quality claim.

## How to read the numbers

**Mmap has no `FileReader` I/O.** `--io-trace` needs `--moe-stream`. Under
plain mmap, storage shows up as **major faults** charged to the graph node
that touched the missing page — almost always `MUL_MAT_ID` (expert weights).
The engine cannot split one node's wall into “ALU vs disk”. Compare a
zero-fault host, or the same host after the working set stayed resident.

Buckets from v2 `--compute-trace` decode frames (`phase=1`), interval union
aligned with `scripts/decode-analyze.py timeline`:

- Expert matmul = `MUL_MAT_ID` union
- Other graph = remaining graph ops
- Callback = `BMOE_CALLBACK`
- Other = decode frame minus graph minus callback
- Weight load (FileReader union) = 0 unless `--moe-stream`

Hot = session turn 1 (0-based). Turn 0 is discarded warmup for the
**headline**. Cold turn 0 is in an appendix because it was recorded.

Before each mmap session the model file got `POSIX_FADV_DONTNEED` (that file
only; no global `drop_caches`). Two `generate` requests, same prompt,
`clear_kv=true`. `--ubatch 128` on A/B, `64` on C.

## Headline: hot mmap

No `--moe-stream`. `o_direct=0`. `--compute-trace` on. 32 decode tokens.

### Wall split (ms/token)

| Component | Host A ~30 GiB | Host B 7.6 GiB | Host C 3.6 GiB |
|---|---:|---:|---:|
| Expert matmul | 104.5 (42.8%) | 114.1 (41.1%) | 734.5 (64.7%) |
| Other graph | 115.7 (47.4%) | 138.2 (49.7%) | 362.1 (31.9%) |
| Callback | 16.1 (6.6%) | 17.1 (6.2%) | 25.8 (2.3%) |
| Other | 7.7 (3.2%) | 8.4 (3.0%) | 12.7 (1.1%) |
| **Decode wall** | **244.0** | **277.9** | **1135.2** |
| CLI tok/s | 4.10 | 3.60 | 0.88 |

Host A is the compute floor for this SoC, quant, thread count and trace:
`MUL_MAT_ID` ≈ 104 ms with **zero** major faults.

### Mmap I/O signal (not FileReader)

| | Host A | Host B | Host C |
|---|---:|---:|---:|
| Engine expert read bytes | 0 | 0 | 0 |
| majflt / token | **0** | 2.1 | **7123** |
| Wall in nodes that had ≥1 majflt | 0 | 0.3 ms | **738 ms** |
| File-backed RSS (end of hot turn) | 8446 MiB (whole file) | 6851 MiB | 2662 MiB |
| Anon RSS | 58 MiB | 39 MiB | 4.8 MiB |
| Swap | 0 | 7.4 MiB | 41 MiB |
| Extra decode vs Host A hot | — | +34 ms | **+891 ms** |

Faults on B and C land on `MUL_MAT_ID` (experts). On C, dense `MUL_MAT`
also takes ~130 majflt/token: the kernel does **not** pin non-expert
weights.

### What this shows

Ordinary mmap does **not** distinguish MoE experts from dense tensors. The
mapping is one file. The kernel LRU is by page. Non-expert weights stay
resident only because every token touches them — until pressure reclaims
them too (Host C). Unused experts are never faulted in. Used expert pages
compete with everything else.

- **A (~30 GiB):** the 8.45 GiB file stays mapped. Hot decode is compute.
- **B (7.6 GiB):** cannot hold the whole file. After this prompt's working
  set warmed, majflt dropped to ~2/token and matmul returned near the
  104 ms floor (114 ms).
- **C (3.6 GiB):** cannot hold the working set. Hot is as slow as cold.
  ~65% of decode wall is nodes that are faulting. The extra ~890 ms/token
  versus A is mmap I/O hidden inside `MUL_MAT_ID`, not 7× more arithmetic.

## Appendix A — cold mmap (same sessions, turn 0)

Recorded, not the headline.

| | A | B | C |
|---|---:|---:|---:|
| Decode wall ms/token | 249.9 | 429.5 | 1147.2 |
| Expert matmul | 104.4 | 233.9 | 745.0 |
| Other graph | 120.9 | 166.2 | 363.8 |
| majflt/token | 0 | 820 | 7361 |
| CLI tok/s | 4.00 | 2.33 | 0.87 |

Host A is already resident after load (30 GiB). Host B's first turn is
still filling expert pages (820 majflt/token, almost all on `MUL_MAT_ID`).
Host C never leaves that regime.

## Appendix B — streaming `O_DIRECT`, cache off, Q4_0

`--moe-stream --cache-mb 0 --dense-weights anon --io-threads 2`, same
prompt and `-n 32`, `--compute-trace` + `--io-trace`. **Not mmap.** Expert
misses use `O_DIRECT`. Shared slots overwrite each layer; extra RAM does
not retain experts. Run so that FileReader breakdown exists; it does not
test page-cache capacity.

| | A | B | C |
|---|---:|---:|---:|
| Decode wall ms/token | 473.3 | 504.7 | 507.0 |
| Expert matmul | 21.7% | 21.9% | 20.6% |
| Other graph | 26.1% | 27.7% | 27.9% |
| Weight load (read-busy union) | 46.4% | 44.5% | 45.8% |
| Callback (contains serial load) | 50.4% | 48.6% | 49.7% |
| Other | 1.8% | 1.8% | 1.8% |
| CLI tok/s | 2.11 | 1.98 | 1.97 |
| Read | 725.77 MiB/token, 468 reads/frame | same | same |
| `o_direct` | 1 | 1 | 1 |
| Dense anon | 715 MiB | 715 MiB | 715 MiB |

Serial path: Weight load ⊂ Callback; do not add those rows. Three hosts
match because this arm measures NVMe + Carmel, not `mem=`. That is why
RAM looked irrelevant until the mmap arm.

An earlier Q4_K_M O_DIRECT cache-off cell on B/C read **853.6 MiB/token**
(mixed down-proj types). Same shape, larger slices.

## Appendix C — local x86 identity (Q4_0)

CPU-only host, same Q4_0 file, 32 greedy tokens:

- mmap vs `--moe-stream --cache-mb 0 --dense-weights anon`: **equal generated text**
- Stream: 725.77 MiB/token, `o_direct=1`

## Portable mmap recipe

```bash
# After POSIX_FADV_DONTNEED on the Q4_0 file (that inode only):
printf '%s\n' \
  '{"cmd":"generate","id":1,"prompt":"The capital of Japan is","n_predict":32,"clear_kv":true}' \
  '{"cmd":"generate","id":2,"prompt":"The capital of Japan is","n_predict":32,"clear_kv":true}' \
| CUDA_VISIBLE_DEVICES= build/cli/bmoe-cli --session \
    -m DeepSeek-V2-Lite-Q4_0.gguf \
    -t 4 -c 256 --ubatch 128 \
    --compute-trace compute.csv --csv metrics.csv
```

Do **not** pass `--moe-stream`. Turn 0 is cold; turn 1 is hot. There is no
`--io-trace` on this path. Use compute-trace `majflt` and RSS
`rss_file_mib` for the time split above. `majflt × page size` is a lower
bound on bytes, not the SSD transfer. The byte measurement is the next
section, and it leaves `--compute-trace` off. On a 4 GiB-visible host use
`--ubatch 64`.

## Interpretation limits

- Traced. Node barriers inflate wall time. Compare shares and majflt,
  not these tok/s to an untraced phone table.
- Hot mmap on B is hot for **this** 32-token greedy trajectory, not for
  long or diverse text. Unique `(layer, expert)` coverage was not re-measured
  on Q4_0 beyond the streamer's 725.77 MiB/token demand.
- mmap will not keep non-expert weights by policy. Host C dense `MUL_MAT`
  still faults.
- `--cache-mb auto` was not run on these boards (it would size from
  `MemAvailable` toward the 7722 MiB bank and can OOM on the 4 GiB cap).
- Taskset/NUMA were not applied; 8 cores are the whole package.

Local gitignored CSVs on the boards (not in this repo):
`build/measured-mmap-q4_0/`, `build/measured-io-breakdown-q4_0/`.

## Block-layer bytes, 2026-09-22

Same three boards, same Q4_0 file, same prompt, `-n 32 -c 256 -t 4`,
greedy, no `--moe-stream`, no `--compute-trace`. `--ubatch 128` on A/B,
`64` on C. The model file was `POSIX_FADV_DONTNEED`'d before each
process. Two `generate` calls, `clear_kv=true`. Turn 0 is cold, turn 1
is hot.

`block_read_*` is the process `/proc/self/io` `read_bytes` delta: bytes
submitted to the block layer, readahead included, cache hits excluded.
Zram swap-in is in that number. A side sample of `nvme0n1` read sectors
over the same window matches it; zram is the small remainder below.
These fields are not in `416b7a9`. The repeat procedure, including which
number is a prefill total and which is per generated token, is in
[AGENTS.md](../../AGENTS.md) under "Mmap block-layer traffic".

### Load (`open`, before the first generate)

| | Host A | Host B | Host C |
|---|---:|---:|---:|
| Process block read | 8441.02 MiB | 8449.68 | 8450.55 |
| NVMe over the same window | 8441.02 MiB | 8456.36 | 8504.54 |
| Zram read | 0 | 0 | 10.18 MiB |
| Load wall | 5.66 s | 9.70 s | 12.89 s |

8441.02 MiB is the Q4_0 file. All three caps read the whole file once
at load. The cap shows up afterwards, as re-reads.

### Decode block read, MiB/token

Engine delta around each `llama_decode`. 32 tokens. `read_mib` (FileReader)
is 0 on every row.

| | Host A | Host B | Host C |
|---|---:|---:|---:|
| Cold, mean | 0.00 | 47.24 | 443.03 |
| Cold, min–max | 0–0 | 0.00–148.83 | 166.59–1149.09 |
| Cold majflt/token | 0 | 876.28 | 7420.97 |
| Cold tok/s | 5.13 | 2.58 | 0.91 |
| **Hot, mean** | **0.00** | **0.36** | **426.25** |
| Hot, min–max | 0–0 | 0.00–5.80 | 161.77–696.51 |
| Hot majflt/token | 0 | 6.66 | 7134.78 |
| Hot tok/s | 5.12 | 4.55 | 0.92 |
| Hot prefill total (6 prompt tokens) | 0 | 85.03 MiB | 3063.94 MiB |
| File RSS at end of hot | 8446 MiB | 6748 MiB | 2643 MiB |
| Swap at end of hot | 0 | 16.5 MiB | 38.8 MiB |

Hot-turn window, prefill plus 32 decode tokens, device counters:

| | Process | NVMe | Zram |
|---|---:|---:|---:|
| A | 0 | 0 | 0 |
| B | 96.48 MiB | 101.45 | 4.67 |
| C | 16709 MiB | 16931 | 105.61 |

C's hot decode is 426 MiB/token against a logical expert touch of
725.77 MiB/token, and against a fault floor of about 27.9 MiB/token
(7135 × 4 KiB). The pages still inside the 2643 MiB file RSS are not
read again. A and B, once this trajectory is resident, add essentially
no SSD traffic. These tok/s are untraced; do not set them next to the
traced table above. C's hot majflt (7135) is the same regime as that
table's 7123.
