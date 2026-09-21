# Host CPU profiling findings

This is a **host Linux CPU / cgroup-v2 retrospective**. It is not a phone
performance table and it does not replace [benchmarks.md](benchmarks.md) or
[benchmark-method.md](benchmark-method.md). Device numbers in those documents
are unchanged.

The main profiling and cgroup runs measured behaviour on an x86 Linux host with **CUDA off**,
**32 compute threads**, CPU affinity `0-31`, **serial** `--moe-stream` (no
`--overlap` unless a smoke control says otherwise), `--dense-weights anon`, and
`--io-threads 2` on the scaled and DeepSeek runs. Greedy decoding, prompt
`The capital of Japan is`, `-n 32`, `-c 256`. GPU load is disabled in
`Session::open` (`n_gpu_layers=0`).

Local `build/` trees named in this page are **optional, gitignored evidence**.
They are not portable Markdown links; a clean clone does not need them. The
tables here are the durable record.

## Snapshot

| Item | Value |
|---|---|
| Code checkpoint | `9f9315e508897d3d329aebd1da67180583754588` |
| Engine version reported by the binary / metrics CSV | `0.23.0` (`CMakeLists.txt` `project(... VERSION 0.23.0)`) |
| Changelog at the same snapshot | latest heading is `[0.24.0]` |
| llama.cpp pin | `0e8c83e512b99ccf83e50798b86f0b0ec40a7b0a` (fork branch with the optional expert-ready hook) |
| Host build | `JOBS=16 bash scripts/build-host.sh`, Release, CUDA false |

The `0.23.0` engine versus a changelog that already begins at `0.24.0` is a
**snapshot discrepancy**. This document does not bump versions.

`run()` is a one-shot wrapper over `Session`. `Session::open` loads with
`LLAMA_LOAD_MODE_MMAP`, `use_extra_bufts=false`, `n_gpu_layers=0`. External
`RouterHook` / `FileReader` / cache code rebind `tensor->data`. Normal serial
streaming uses public llama.cpp APIs; optional `--overlap` needs the internal
CPU ready hook in the pinned fork. Expert read/cache management stays outside
llama.cpp; initial model loading still goes through llama.cpp.

## How to read these numbers

**A traced run is not a throughput benchmark.** `--compute-trace` isolates every
graph node (no ggml operator coalescing). `--io-trace` takes a lock per read.
Read proportions, not absolute tok/s, from traced cells. Untraced controls are
reported separately where they exist.

Tables round statistics to the displayed precision; byte counts are exact unless explicitly
labelled approximate. Aggregate calculations use the recorded, unrounded values.

**Wall intervals are not pure arithmetic.**

- Per-node graph wall includes backend synchronization, scheduling, and expert
  readiness waits.
- FileReader intervals include copy and alignment, not device-only service time.
- Parallel reads and waits are **unioned** for wall metrics. `read_busy_sum_ms`
  adds per-read durations across lanes; it is not a disjoint share of total wall time.
- `taskset -c 0-31` is affinity, not CPU exclusivity. Other processes can still
  run on those CPUs. Contention can distort wall-time *attribution*. It cannot
  turn actual read bytes or read events into zero.

**Zero hot expert I/O is not a routing-locality result.** The warm and cgroup
cells repeat the **same** 32-token greedy request in one process with
`clear_kv=true`. That drops KV and conversation state but **retains expert
weights** and **replays the same routing**. Zero expert I/O after warmup means
the working set for *this* request stayed resident. It is **not** evidence that
adjacent tokens of new or long text change few experts, and it is **not** a
measurement of long-generation I/O.

**`--cache-mb auto` is a frozen budget, not a runtime governor.** It reads host
`/proc/meminfo` `MemAvailable` once at init. It subtracts pending anonymous /
pinned dense allocations (and a row-stream window when that policy applies). It
does **not** read cgroup `memory.max` or `memory.current`. Under a memory cap
the selected budget can exceed the cgroup.

**O_DIRECT is not end-to-end page-cache avoidance.** Linux expert misses request
`O_DIRECT`, with buffered fallback, a verify comparison block, and a
sub-alignment EOF tail. Effective mode is what telemetry prints as `o_direct`.
`--no-odirect` changes **expert** readers only. Dense `anon` uses its own
direct `FileReader`, independent of that flag. Initial mmap / capture warmup
still uses the page cache: a 0.23.0 Linux strace on this pin observed
`MAP_SHARED|MAP_POPULATE` and `MADV_WILLNEED` before successful `O_DIRECT`
`pread`s.

**Memory figures on this kernel are snapshots.** `memory.peak` was not
available. `memory.current` is sampled, not an exact peak. Tracing output can
charge page cache to the cgroup.

## Already run versus not run

### Already run

1. Random tiny qwen3moe / gemma4 fixtures: execution and **byte-identity**
   smoke, including an `O_DIRECT` path and one strace. Not quality, not
   quantization, not over-RAM performance.
2. Scaled random-F32 qwen3moe (~818 MiB file, 768 MiB expert bank) with a
   **128 MiB** expert cache (`--force-cache`): traced and untraced serial, plus
   overlap controls. I/O-dominated; **misleading** if read as a warm or
   full-capacity result.
3. Same scaled model, **same process**, `--cache-mb auto` (768 MiB, full expert
   bank): one discarded warmup generation, three measured identical greedy
   32-token requests, `clear_kv=true`. Hot expert I/O went to zero.
4. Same warm synthetic workload inside cgroup-v2 caps of 4 / 8 / 16 / 32 GiB,
   both ascending and descending order, `MemorySwapMax=0`. Caps were
   **non-binding** (~1.60 GiB charged). Hot reads, major faults, swap, and
   OOM counters stayed zero.
5. Real **DeepSeek-V2-Lite** Hugging Face → F16 GGUF → native `Q4_K_M`, then
   the same cgroup matrix. 4 GiB **OOM during excluded warmup**. 8 / 16 / 32 GiB
   completed; hot decode I/O was zero for this repeated request. Resident vs
   streamed output was byte-identical on a 32-token smoke.

Normal, non-cancelled traced runs in those cells were validated with
`scripts/decode-analyze.py timeline`.

### Not run (paused)

- Novel **long-generation** or **route-change** text (different prompts, long
  continuations, `clear_kv=false` chat that actually walks new tokens).
- A CPU-contention study that would isolate wall-time distortion (exclusivity,
  known co-runners, attribution vs byte counts).
- Reproducing the cancel/trace-flush bug with a dedicated cancel regression.
- Phone / UFS re-measurement; nothing here updates device benchmark tables.
- Further conversion, quantization, or inference. This page is documentation of
  work that already happened.

## Known issue: cancel can leave a compute-trace frame open

**Status: static finding in source. Not reproduced by a cancel regression. Not
fixed.**

`RouterHook::begin_compute_batch` emits a `BMOE_DECODE` row with `end_ns=0`.
`Session::generate` closes it in `trace_flush()` after a successful
`llama_decode`. Prefill error/cancel **returns** before that flush. Decode
error **returns** before it; decode cancel **breaks** the loop and then rolls
the turn back, still without flushing that frame.

A later successful generation in the same process may flush the old invalid
row. `scripts/decode-analyze.py timeline` refuses inverted spans
(`start_ns > end_ns`). Successful non-cancelled runs in this investigation did
not hit that path.

## 1. Tiny random fixture proof

**Scope.** Repository generator `scripts/make-tiny-moe.py`. Random F32 weights.
qwen3moe fixture 7,375,136 bytes; gemma4 fixture 7,250,400 bytes. Prompt as
above, `-t 4 -c 256 -n 32`. Engine `0.23.0`, llama.cpp pin as in the snapshot
table. No source modifications.

**What it proves.** Streamed output matched the mmap-resident control
(`matches_mmap_generated_bytes: true`) on qwen3moe direct, buffered, LRU, and
overlap controls, and on gemma4 mmap / direct. Direct qwen3moe used
`--moe-stream --cache-mb 0 --dense-weights anon`. A strace of that path
recorded the initial mmap and later expert / dense direct opens.

**What it does not prove.** Model quality, production quantizations, over-RAM
behaviour, or host tok/s. Do not quote the tiny-fixture rates as performance.

## 2. Misleading 128 MiB-cache scaled F32 run

**Workload.** Same generator family (`build_qwen3moe`) with a larger random-F32
shape, **not** the stock `make-tiny-moe.py` CLI defaults:

| Field | Value |
|---|---|
| Architecture | qwen3moe, 4 layers, 16 experts, top-2 |
| Width | `n_embd=1024`, `n_head=16`, `n_head_kv=8`, `n_ff=4096`, `n_ff_exp=1024`, `n_ctx=256` |
| File | 858,070,816 bytes (~818 MiB) |
| Expert bank | 805,306,368 bytes (768 MiB) |
| Cache | **128 MiB** with `--force-cache` (below `cache_min_mb`, far below the 768 MiB bank) |
| Threads / lanes | 32 / 2 |
| Direct I/O requested | yes (`o_direct=1` on the streaming runs) |
| GPU | disabled in build and load |

Weights are deterministic random F32. No pretrained quality claim.

The 128 MiB cap cannot hold the expert bank. Decode read 54.43 MiB/token over 32
tokens. Run-level counters reported 37.1% cache hit, 178 evictions and 126 re-reads;
those counters include earlier activity such as prefill. This is deliberately
restricted-cache traffic, not full-capacity hot replay. It is not the zero-hit
cache cliff: the configured budget exceeded this fixture's 96 MiB token-cycle estimate.

### Untraced controls (median of 3 runs × 32 tokens)

Prefill excluded from the per-token medians below. `cpu_ms` is process CPU time
(all threads), not wall.

| Mode | tok/s median | wall ms/token median | io_ms | stall_ms | mgmt_ms | compute_ms (residual) |
|---|---:|---:|---:|---:|---:|---:|
| serial | 23.707 | 42.182 | 33.792 | 0 | 3.383 | 4.737 |
| overlap | 17.807 | 56.157 | 11.607 | 21.985 | 7.129 | 26.455 |

The untraced serial blocked-load phase was about 80% of wall time. That includes
reader work and scheduling, not storage-device-only service. Overlap on the same
restricted cache was slower in these controls; this is not a universal overlap result.

### Per-node traced aggregates (32 decode frames, prefill excluded)

| Mode | decode wall ms | graph wall ms | expert matmul ms | callback ms | read-busy union ms | read-busy sum ms | read bytes | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| serial | 1591.480 | 380.705 | 126.505 | 1199.869 | 1068.176 | 1741.662 | 1,826,303,328 | 32 |
| overlap | 1938.498 | 1638.599 | 1024.578 | 287.242 | 1117.445 | 1823.580 | 1,826,303,328 | 32 |

Serial traced decode averaged 49.734 ms/token. Its 37.496 ms/token callback interval
**includes** the 33.380 ms/token reader-service union, plus about 4.115 ms/token
outside those read intervals. It is not 37.496 ms of profiler overhead.
Graph-node elapsed was 11.897 ms/token and other frame work 0.341 ms/token.
Tracing changes the schedule; the difference from the untraced control must not be
attributed wholesale to callback instrumentation. Generated bytes matched across the runs.

## 3. Corrected full-capacity same-process warm replay

**Same scaled F32 model**, still serial, 32 threads, 2 lanes, greedy 32-token
prompt, `clear_kv=true`. Cache **`--cache-mb auto`**: effective budget **768 MiB**
(the full expert bank). After the discarded warmup, **744 MiB** of experts were
resident — the working set this request touched, not a claim that every expert
in the file was used.

One process, four turns; **turn 0 discarded**; turns 1–3 measured (**96** hot
decode tokens / frames). Decode-only.

| Quantity | Hot aggregate (96 frames) | Per hot token |
|---|---:|---:|
| decode wall | 555.213 ms | 5.783 ms |
| graph wall | 476.384 ms | 4.962 ms |
| expert matmul (`MUL_MAT_ID`) | 126.043 ms | 1.313 ms |
| other graph | 350.341 ms | 3.649 ms |
| callback | 52.418 ms | 0.546 ms |
| other | 26.412 ms | 0.275 ms |
| expert wait union | 0 | 0 |
| read-busy union / sum | 0 / 0 | 0 |
| read bytes | **0** | **0** |

I/O trace events: turn 0 = 186; turns 1, 2, 3 = **0**.

Section 2 included first-touch misses and repeated capacity misses caused by the
128 MiB cap. It was not a fully resident hot baseline. With enough expert-cache
capacity, the repeated request required **no expert reads** after warmup. This still
does **not** test new text or long generation: the measured turns replayed the same
routing. The cumulative cache-hit percentage can retain earlier cold misses even
when a later turn has zero reads; inspect turn-filtered byte/event counts.

## 4. Synthetic cgroup-v2 caps (non-binding)

**Same warm synthetic workload** as section 3, now inside a user cgroup with
`MemoryMax` ∈ {4, 8, 16, 32} GiB and `MemorySwapMax=0`. Two run orders
(4→32 and 32→4). One warmup generation discarded per session; three measured
generations; **192 hot tokens per cap** (2 sessions × 3 × 32). Cache auto still
sized to the **768 MiB** expert set. Caps were verified from kernel files. The
model file was charged **inside** each cgroup (per-file `POSIX_FADV_DONTNEED`
before a fresh service; no global `drop_caches`).

Charged usage after measurement was **~1644 MiB (~1.60 GiB)** including anon, file
and kernel memory. This is not a minimum required working set. All tested caps were above it,
and were **non-binding**. Hot expert I/O, major faults, swap, memory pressure events, and
OOM counters were **zero**. Timing scatter is **not** evidence of a memory-limit
bottleneck.

Traced hot decode (prefill excluded). `memory.current` is a snapshot, not a peak.

| Cap (GiB) | Hot tokens | Mean ms/token | Median generation mean | Generation-mean range |
|---:|---:|---:|---:|---|
| 4 | 192 | 4.489 | 4.474 | 3.301–5.729 ms/token |
| 8 | 192 | 8.283 | 5.679 | 3.297–24.976 ms/token |
| 16 | 192 | 4.647 | 4.680 | 3.344–6.144 ms/token |
| 32 | 192 | 4.271 | 3.785 | 3.352–5.677 ms/token |

Breakdown below is mean ms/token. Every cell had zero read bytes, swap and
`memory.events` high/max/oom/oom_kill counts. The memory column is the larger
end-of-measurement snapshot from the two sessions, **not** a peak.

| Cap (GiB) | Expert matmul | Other graph | Weight load | Callback | Other | Snapshot MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.874 | 2.841 | 0 | 0.504 | 0.269 | 1643.7 |
| 8 | 1.508 | 5.993 | 0 | 0.511 | 0.271 | 1643.7 |
| 16 | 0.937 | 2.887 | 0 | 0.542 | 0.281 | 1643.5 |
| 32 | 0.792 | 2.648 | 0 | 0.544 | 0.287 | 1644.3 |

The 8 GiB **mean** is pulled by a slow generation (range up to ~25 ms/token);
the median stays in the same band as the other caps. Transient units were
stopped after the measurements.

## 5. DeepSeek-V2-Lite conversion and cgroup matrix

### Model and files

Architecture `deepseek2` (recipe: standard split `ffn_{gate,up,down}_exps`).
Leading dense FFN, shared experts, and MLA stay on the dense-weight policy.

| Field | Value |
|---|---|
| Layers | 27 total, **1** leading dense → **26** MoE |
| Routed experts | 64 per MoE layer, top-6 (`n_expert_used=6`) |
| Conversion | pinned `convert_hf_to_gguf.py`, `--outtype f16`, no GPU |
| F16 GGUF | 31,424,035,840 bytes (**29.266 GiB**) |
| Quantization | llama.cpp native `Q4_K_M` from that F16, 32 CPU threads |
| Q4_K_M GGUF | 10,367,958,016 bytes (**9.656 GiB**) |
| Tensors | 404 total; native mixed fallback **54 / 404** |
| Reported average bpw | 5.28 |
| Routed tensors | 78 (types `Q4_K`, `Q5_0`, `Q8_0`) |
| Routed bank | 9,527,361,536 bytes (9086 MiB) |
| Non-routed bytes in the Q4 file | 836,597,760 |
| HF / llama.cpp sources modified | no |

Resident vs streamed 32-token greedy smoke: both exit 0,
`equal_generated_bytes: true`.

### Cgroup experiment

Same serial recipe as the synthetic caps: 32 threads, affinity `0-31`, 2 I/O
lanes, `--overlap` off, `--dense-weights anon`, `--cache-mb auto` **with no
manual reduction**, `n_ctx=256`, 1 warmup + 3 measured requests of 32 tokens,
`clear_kv=true`, paired per-node compute and I/O traces, hot decode only.
`MemorySwapMax=0`. If a cap killed the process before warmup finished, the
cell is **OOM** with **no** hot breakdown — cache policy was not quietly
shrunk and the model was not swapped for a smaller one.

`--cache-mb auto` still read **host** `MemAvailable`, so every cell selected
**9086 MiB** (the full routed bank) even under a 4 GiB cgroup. That is a
property of the auto budget, not a mistake in the table.

This repeated request warmed **6859 MiB** of routed experts, not all 9086 MiB.

### Hot traced decode

Denominator for completed cells: **96** hot tokens (3 × 32), warmup excluded.
4 GiB: warmup OOM, **0** hot tokens, every timing/IO column **N/A**.
`memory.current` is a snapshot. 8 GiB and 16 GiB sat at the cap by reclaiming
clean mapped-file pages; 32 GiB did not fill the cap.

Times are mean ms/token; completed columns each contain 96 hot tokens.

| Component | 4 GiB | 8 GiB | 16 GiB | 32 GiB |
|---|---:|---:|---:|---:|
| Expert matmul | N/A | 7.032 | 6.974 | 6.931 |
| Other graph | N/A | 19.493 | 19.417 | 19.507 |
| Weight load | N/A | 0 | 0 | 0 |
| Callback | N/A | 3.355 | 3.347 | 3.290 |
| Other | N/A | 1.701 | 1.674 | 1.653 |
| Total | Warmup OOM | 31.580 | 31.412 | 31.381 |

Completed cases had zero hot expert read bytes and zero hot major faults.
The common cache budget was 9086 MiB; resident experts were 6859 MiB after warmup.
Memory snapshots include reclaimable file pages and tracing output, not just needed weights.

| Cap (GiB) | Outcome | Snapshot GiB | Anon GiB | File GiB |
|---:|---|---:|---:|---:|
| 4 | Warmup OOM, no hot samples | N/A | N/A | N/A |
| 8 | Complete | 8.000 | 7.534 | 0.416 |
| 16 | Complete | 16.000 | 7.533 | 8.405 |
| 32 | Complete | 17.300 | 7.533 | 9.704 |

4 GiB reached `BMOE_READY` with `memory.current` already against the 4 GiB max,
then systemd `Result=oom-kill` on request 1 (no `BMOE_DONE`). Capture that
result **before** tearing the unit down. Hot traces for 8 / 16 / 32 GiB show
zero explicit expert reads and zero major faults in the measured window.

ms/token here includes trace isolation, sync, scheduling, and callback time.
It is not untraced throughput and not a phone number.

Zero hot I/O again means **this** 32-token continuation was served from the
6859 MiB already-resident expert set. It does not measure DeepSeek routing
churn on long or new text.

## Portable recipes

Placeholders: `MODEL_SOURCE` (Hugging Face tree), `GGUF_DIR`, `MODEL_GGUF`.
Do not paste machine-local paths into reports.

### Build

```bash
git submodule update --init --recursive
JOBS=16 bash scripts/build-host.sh
# binary: build/cli/bmoe-cli
```

The recorded host build had CUDA and other GPU backends disabled, and
`Session::open` used `n_gpu_layers=0`. Check build settings and runtime layer
placement rather than assuming an inherited CMake cache is CPU-only.

### Tiny fixture smoke (identity, not performance)

NumPy is required; use the pinned submodule's Python package for `gguf` if it
is not installed in the active environment:

```bash
PYTHONPATH="$PWD/third_party/llama.cpp/gguf-py" \
  python3 scripts/make-tiny-moe.py --arch qwen3moe --out tiny-qwen3moe.gguf
PYTHONPATH="$PWD/third_party/llama.cpp/gguf-py" \
  python3 scripts/make-tiny-moe.py --arch gemma4 --out tiny-gemma4.gguf

build/cli/bmoe-cli -m tiny-qwen3moe.gguf -p "The capital of Japan is" \
  -t 4 -c 256 -n 32 --progress --csv qwen-mmap.csv
build/cli/bmoe-cli -m tiny-qwen3moe.gguf --moe-stream --cache-mb 0 \
  --dense-weights anon --io-threads 4 -t 4 -c 256 -n 32 \
  -p "The capital of Japan is" --progress --csv qwen-direct.csv
```

Compare generated bytes, not telemetry or timings. The scaled F32 shape in
section 2 used the same generator with in-memory overrides; the stock CLI
does not accept that shape as extra flags.

### Conversion / quantization (DeepSeek-V2-Lite, same pin)

Set `MODEL_SOURCE`, `GGUF_DIR` and later `MODEL_GGUF` to your chosen paths.
Enable the native tools target explicitly: llama.cpp is a subdirectory here,
and `LLAMA_BUILD_TOOLS` need not be enabled by its standalone defaults.

```bash
cmake -S . -B build -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-quantize -j 16
CUDA_VISIBLE_DEVICES= python3 third_party/llama.cpp/convert_hf_to_gguf.py \
  "$MODEL_SOURCE" --outfile "$GGUF_DIR/DeepSeek-V2-Lite-F16.gguf" --outtype f16
CUDA_VISIBLE_DEVICES= taskset -c 0-31 build/bin/llama-quantize \
  "$GGUF_DIR/DeepSeek-V2-Lite-F16.gguf" \
  "$GGUF_DIR/DeepSeek-V2-Lite-Q4_K_M.gguf" Q4_K_M 32
```

Choose a valid affinity mask for the host; `0-31` describes the recorded setup,
not a portable guarantee of 32 physical cores.

### Same-process warm profiling

Save these four lines as `requests.jsonl`. End the input after the final line;
EOF lets the CLI process the queued requests and finish its reader.

```jsonl
{"cmd":"generate","id":1,"prompt":"The capital of Japan is","n_predict":32,"clear_kv":true}
{"cmd":"generate","id":2,"prompt":"The capital of Japan is","n_predict":32,"clear_kv":true}
{"cmd":"generate","id":3,"prompt":"The capital of Japan is","n_predict":32,"clear_kv":true}
{"cmd":"generate","id":4,"prompt":"The capital of Japan is","n_predict":32,"clear_kv":true}
```

```bash
CUDA_VISIBLE_DEVICES= taskset -c 0-31 build/cli/bmoe-cli \
  --session --moe-stream --cache-mb auto --dense-weights anon \
  --io-threads 2 -t 32 -c 256 -m "$MODEL_GGUF" \
  --compute-trace compute.csv --io-trace io.csv --csv metrics.csv \
  < requests.jsonl
```

Request id 1 is warmup (trace turn 0). Select turns 1, 2 and 3 explicitly;
without `--turn`, the analysis includes the cold turn as well.

```bash
for turn in 1 2 3; do
  python3 scripts/decode-analyze.py timeline compute.csv --io io.csv \
    --turn "$turn" --phase 1 --svg "hot-turn-$turn.svg" \
    --summary "hot-turn-$turn.json"
done
```

Each command summarizes one 32-token hot turn. For a 96-token combined
result, sum durations and frame/token counts across the three reports before
dividing; do not average percentages with different wall-time denominators.

### Cgroup-v2 cap (user manager)

Requires a user manager that delegates memory. Keep cache policy unchanged
when comparing caps and report OOM rather than silently substituting another
configuration. Use absolute model paths; the working-directory option below
sets where the service resolves output paths.

```bash
systemd-run --user --wait --pipe --unit=bmoe-cap \
  --working-directory="$PWD" \
  -p MemoryAccounting=yes -p MemoryMax=8G -p MemorySwapMax=0 \
  --setenv=CUDA_VISIBLE_DEVICES= \
  taskset -c 0-31 "$PWD/build/cli/bmoe-cli" \
    --session --moe-stream --cache-mb auto --dense-weights anon \
    --io-threads 2 -t 32 -c 256 -m "$MODEL_GGUF" \
    --compute-trace compute.csv --io-trace io.csv --csv metrics.csv \
    < requests.jsonl
```

Repeat with fresh groups for the requested limits. The experimental driver
kept stdin open while taking snapshots between generations, then stopped the
unit after the last `BMOE_DONE`. The finite-input example is the simpler replay;
to reproduce the memory snapshots, observe the live group before it exits.
Do not use automatic collection when it would remove failure evidence before
you record the unit's `Result`, exit status and available memory counters.

## Safeguards (cgroup memory cells)

1. **Verify the real cgroup**, not just launch flags: `MainPID`, that PID's
   `/proc/<pid>/cgroup`, and the kernel `memory.max` / `memory.swap.max` files
   inside that cgroup.
2. **Experiment-owned model file only:** `POSIX_FADV_DONTNEED` on `MODEL_GGUF`
   before each fresh capped service so leftover file pages are not still
   charged to a previous cgroup. Never global `drop_caches`.
3. **No silent cache-budget change** and no substitute smaller model on OOM.
4. **Warm inside the same process**, discard that generation, then measure
   identical prompts with KV cleared and the expert cache retained — if that is
   the question. A different question (new text, long generation) needs a
   different protocol.
5. **Snapshots, not peaks** on kernels without `memory.peak`. Record
   `memory.current`, `memory.stat` (anon/file), `memory.events`,
   `memory.pressure`, and swap.
6. **Capture OOM before cleanup.** systemd `Result=oom-kill` (and
   `memory.events`) must be recorded while they still exist.
7. **Stop the actual transient unit.** Killing only the `systemd-run` wrapper
   is not enough; `systemctl --user stop <unit>` (or equivalent) the service
   that holds `MemoryMax`.

## Optional local evidence

On a machine that already ran this work, gitignored trees under `build/` hold
the raw CSVs, cgroup snapshots, and JSON summaries (`build/smoke/`,
`build/measured-cpu32/`, `build/measured-cpu32-warm/`,
`build/measured-cgroup-memory/`, `build/measured-deepseek-cgroup/`). They are
ignored for a reason: they contain host paths and are not part of a clean
clone. Do not copy them into `docs/`.
