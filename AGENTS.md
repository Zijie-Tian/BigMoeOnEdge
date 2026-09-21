# Working on BigMoeOnEdge (agent guide)

Read this before making changes. It captures the invariants that keep this project clean.
It follows the [AGENTS.md](https://agents.md) convention, so any coding agent picks it up;
`CLAUDE.md` just points here.

Quick navigation: [hard rules](#hard-rules), [build](#build-and-test),
[execution/cache](#execution-and-cache-model), [page cache](#page-cache-exact-scope-of-bypass),
[cgroups](#memory-budgets-and-cgroups), [profiling](#traces-and-measurement-semantics),
[measured findings](#what-the-investigation-established), [open questions](#interpretation-limits-and-deferred-work).

## What this is

A ports-and-adapters engine that streams MoE experts from storage so models larger than
physical RAM can run on a device. llama.cpp still owns model loading, tokenization, graphs,
operators and KV state. BigMoeOnEdge owns expert residency, selective reads and cache policy,
in the **same process**, rather than replacing llama.cpp's entire I/O subsystem.

The serial streaming adapter uses public llama/ggml/GGUF APIs and needs no internal patch.
The **pinned submodule is a fork**, however: optional `--overlap` uses its small expert-ready
hook inside the CPU expert kernel. Chat templates, reasoning parsing and MTP also use
llama.cpp's `common` library, which is not a stable public API. Do not describe the whole
checkout as unmodified upstream or every integration as stable.

Read [architecture](docs/architecture.md) and [the seam](docs/seam.md) for the implementation
contract. [CPU profiling findings](docs/cpu-profiling-findings.md) records the investigation,
including unsuccessful configurations and corrections to earlier interpretations.

## Project map

| Location | Responsibility |
|---|---|
| `core/include/bmoe/` | Ports, configuration, session/runtime APIs and telemetry contracts; no llama.cpp dependency in public policy headers. |
| `core/src/io/platform_io.*` | Platform file descriptors, positioned reads, aligned allocation, VM reservation/eviction and memory sensors. |
| `core/src/io/file_reader.*` | Per-file, per-lane reader; direct-mode verification/fallback, alignment, bounce buffers and EOF tails. |
| `core/src/io/mapping_release.*` | Optional release of the model file's mappings after safe rebinding; preserves address placeholders for teardown. |
| `core/src/moe/gguf_offsets.*` | GGUF metadata and tensor-to-shard/byte-offset mapping, without allocating model weights. |
| `core/src/moe/arch_registry.cpp` | Architecture recipes: expert tensor suffixes, not model-specific sizes. |
| `core/src/moe/router_hook.*` | Public eval callback, capture, routing IDs, optional routing policies and graph traces. |
| `core/src/moe/expert_stream_source.*` | Shared slots/LRU, read workers, demand/prefetch jobs and overlap readiness. |
| `core/src/moe/dense_weights.*`, `row_stream.*` | Non-expert residency and optional row-gathered table streaming. |
| `core/src/engine/session.cpp` | Composition root: load once, capture/rebind, prefill/generate repeatedly, cancel and tear down. |
| `core/src/engine/runtime.cpp` | One-shot wrapper over `Session`; not a second generation engine. |
| `core/src/engine/chat_parse.*`, `thinking_control.*`, `ngram_draft.*` | Model-template integration, thinking control and text-based speculative drafting. |
| `core/src/metrics/` | Metrics, route-trace and compute/I/O-trace sinks. |
| `cli/main.cpp` | `bmoe-cli`, argument/env resolution and session JSON protocol; the only project layer that reads configuration environment variables. |
| `third_party/llama.cpp` | Pinned dependency with the optional expert-ready hook; never edit its sources in-tree. |
| `scripts/` | Official build, GGUF fixture generation, trace analysis and benchmark utilities. |
| `tests/` | Byte-identity gates and focused policy/integration tests. |
| `tools/` | Optional storage/memory diagnostic executables, not the inference engine. |
| `examples/android/` | App driving the CLI through a process, not a separate inference implementation. |

## Build and test

Host requirements: CMake 3.21+, a C++17 compiler and the pinned submodule. Python with NumPy
and `gguf` is needed for synthetic-model gates, not for ordinary C++ GGUF inference.

```bash
git submodule update --init --recursive
JOBS=16 bash scripts/build-host.sh
PYTHONPATH="$PWD/third_party/llama.cpp/gguf-py" \
  ctest --test-dir build --output-on-failure
build/cli/bmoe-cli --version
build/cli/bmoe-cli --list-archs
```

The vendored `gguf-py` can supply `gguf` without changing the user's Python environment.
The host script honors `BUILD_DIR`, `BUILD_TYPE` and `JOBS`; it defaults to Release and
produces `build/cli/bmoe-cli`. Without the submodule, top-level CMake can configure only
pure-policy code; that is **not** a working inference build. The host script rejects that case.

Enable the native quantizer explicitly when using llama.cpp as this project's subdirectory:

```bash
cmake -S . -B build -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-quantize -j 16
```

`llama-quantize` otherwise may not exist as a target. Do not mistake that for a compiler failure.
For CPU-only measurements, verify CPU backend/build settings and the runtime's layer placement;
`CUDA_VISIBLE_DEVICES=` alone is not proof that a binary is CPU-only.

Android CLI: `pwsh scripts/build-android.ps1` (needs the NDK), then build the APK in
`examples/android`. Host correctness checks do not replace real-device release validation.

## Hard rules

1. **Never patch llama.cpp in-tree.** The serial adapter uses public eval-callback and
   GGUF/model APIs; the pinned fork's existing expert-ready hook is the documented overlap
   exception. If a new change needs an internal edit, stop and discuss. An agreed extension
   belongs on a separate minimal fork branch, never an in-tree diff. Upgrading llama.cpp must
   stay a submodule bump, with the hook and `common` integration re-verified.
2. **Repack stays off.** Current loading uses `load_mode=LLAMA_LOAD_MODE_MMAP` and
   `use_extra_bufts=false` (older docs called this `use_mmap=true`). The streamer rebinds
   `tensor->data` to native GGUF layout; repacking breaks it. This is not a tuning knob.
3. **No env vars in the library.** `core/` never calls `getenv`. Config flows through
   `RunConfig`; the CLI resolves any env overrides before building it.
4. **No hardcoding.** New architectures are recipe rows in `arch_registry.cpp`; expert
   counts, strides and offsets are discovered at runtime. No model-specific constants in
   the streaming path.
5. **Gates must pass before merge.** `bmoe_moe_gates` proves streamed == resident. If you
   touch the streamer, the seam, or bump the submodule, run them.
6. **Docs and changelog ship with the change.** Every PR updates `CHANGELOG.md` and the docs
   it invalidates, in the same PR — never as a later sweep. A release gets its own dated
   `## [X.Y.Z] - YYYY-MM-DD` section; nothing accumulates under `[Unreleased]`. Check in
   particular: the README benchmark tables and model list, the `docs/architecture.md` layer
   map, `docs/seam.md` when the llama.cpp boundary moves, `docs/telemetry.md` when CSV
   columns or the `BMOE_*` protocol change, `docs/roadmap.md` when a listed future item
   ships, and `examples/android/README.md` when the catalog, settings or build flow change.
   Docs that name a file the code no longer has are worse than no docs.
7. **Review exactly what is being published, every time.** This repo is public and every push
   is permanent record. Before any commit, push, PR or release: run `git status --short` and
   stage by explicit path only — never `git add -A` / `git add .`; untracked files in the
   working tree are not yours to publish. Logs, CSVs and bench evidence get a scan for
   identifying data (device model codes, local paths, addresses) before landing in `docs/`;
   phrase the test device generically. After a squash-merge, verify the landed tree
   (`git ls-tree`) before pushing anything else.

## Conventions

- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `build:`, `ci:`, `chore:`). Author is **Helldez only** — do NOT add AI co-author or
  session trailers to commits.
- **Group commits.** One commit = one coherent change. Never a commit for a trivial
  tweak on its own — fold small fixes, doc touches and follow-ups into the change they
  belong to. If several small things accumulate, batch them into one commit.
- **Delete the branch when its PR closes** — merged or rejected, local and remote
  (`git branch -d`, `git push origin --delete`). A branch list should only show work in
  flight. Nothing is lost on a rejected PR: GitHub keeps its commits reachable from the
  closed PR itself.
- **Language:** all code, comments, docs, and commit messages in English.
- **Style:** `.clang-format` (LLVM base, 4-space, 120 col). CI checks with
  **clang-format 18**; match that version locally (`pip install clang-format==18.*`) or
  formatting that looks clean can still fail the check.
- Comments explain *why* / invariants, not *what*.
- **No milestone codenames** (M0…Mn) in docs — describe capabilities thematically.

## Fork workflow

- The current research branch is `tzj/bitmoe`. The code investigation was checkpointed at
  `9f9315e508897d3d329aebd1da67180583754588`; documentation may evolve after that checkpoint.
- The parent repository intentionally has **only `origin`**, pointing to the user's fork.
  `upstream` was explicitly removed. Do not recreate it or push to another repository without
  a new instruction. This does not change the submodule's own remote or pinned commit.
- Commit/push only when requested, inspect the exact publication scope, and stage explicit
  paths. Models, local traces, generated plots, build output and caches stay out of commits.
- `.gitignore` covers `build*/`, model weights, local benchmark scratch, `.cache/` (including
  clangd indexes), Python bytecode and pytest/mypy/ruff caches. Ignore rules do not untrack
  files already committed; check that distinction before promising a file is excluded.
- Keep one authoritative uppercase `AGENTS.md`; `CLAUDE.md` points here. Do not create a
  competing lowercase `agents.md` on a case-sensitive filesystem.

## Releases

- **Release APKs come from CI, never from a local build.** The `release-apk` workflow runs
  when a release is published: clean checkout of the tag, NDK build, signed with the stable
  key from repository secrets, assets attached to the release. Do not hand-upload an APK.
- **Every released feature bumps the app version**: `versionCode` + `versionName` in the
  Android app's Gradle config, in the same PR as the change being released, matching the tag.
- **Release title is the bare version** — `vX.Y.Z`, no description after it.
- **Validate on device before releasing.** The host gates prove correctness, not speed or
  app behaviour; a release that changes the engine or the app gets a run on a real phone
  first.

## Where numbers come from

Existing Android benchmark figures were measured on a generically described 12 GB / UFS 4.x
phone, using Qwen3-30B-A3B-Q4_K_M and other named models. The host CPU/cgroup findings below
are a separate experiment family, not replacements for those device numbers. Do not invent
or silently round measurements; update `docs/benchmark-method.md` and the relevant README
benchmark table together when re-measuring that published protocol. Keep hardware and paths
generic in public evidence.
The human-readable investigation summaries round to the shown precision; exact byte counts
and unrounded aggregates remain in their recorded manifests. Do not silently change the
precision or measurement scope when copying a value.

## Execution and cache model

### Initialization and the normal decode path

1. `Session::open` loads through llama.cpp with `LLAMA_LOAD_MODE_MMAP`,
   `use_extra_bufts=false` and **`n_gpu_layers=0`**. The current entry point is CPU-only
   for dense weights as well as experts; a GPU in the host does not mean it is used.
2. A capture warm-up walks graph sources and records live expert `ggml_tensor*` objects.
   Their names are matched against the architecture recipe. GGUF metadata supplies shard
   indices and file offsets; expert count and byte stride come from the tensors.
3. The engine allocates its own buffers and rebinds expert `tensor->data`. Native shape and
   stride remain intact: expert `e` is addressed at `tensor_base + e * nb[2]`, and its
   file slice begins at `tensor_file_offset + e * nb[2]`. The existing matmul is retained.
4. For `ffn_moe_topk-<layer>`, the public callback asks the scheduler to compute and synchronize
   the routing node. The non-ask callback gathers the selected IDs and calls `load_layer`.
   Respect the routing view's strides; prefill is not a packed flat array of top-k IDs.
5. In serial mode, missing slices are read and the **whole demand batch is awaited before
   the callback returns**. Multiple read lanes parallelize I/O, not I/O versus this layer's
   expert computation. Already-cached experts can still wait behind this batch barrier.
6. The original llama.cpp graph consumes the rebound weights. Capture KV is cleared before
   real generation. A `Session` can serve multiple requests; `run()` opens one, generates once
   and closes it.

This is external weight-supply/residency management inside the same process, not an IPC
server, a replacement filesystem, SPDK, or a rewrite of llama.cpp's operators.

### What the cache stores

- The cache holds **weights**, not expert outputs or activations. Every token still computes.
- With cache off, one full-size slot per present expert projection is shared across layers.
  Only selected slices are valid; each generation step rereads its selected experts.
- With cache on, buffers are reserved per `(layer, projection)` but LRU entries are keyed by
  **`(layer, expert)`**. An entry accounts for that expert's projections together. Fixed
  virtual addresses preserve the kernel's native indexing; physical pages are filled on demand.
- Hits update LRU state without reads. Eviction releases cold entries' fully contained pages,
  without discarding neighboring slices sharing a page. The currently staged generation is
  protected from eviction. The expert budget is not a process-wide RSS or cgroup limit:
  dense weights, KV, graph buffers, I/O bounces, bookkeeping and file cache also cost memory.
- `GenerateRequest::clear_kv=true` clears KV/conversation state, **not the expert cache**.
  A fresh CLI process starts a fresh expert cache. Use `--session` for a real warm/cold split.

### Overlap is a separate switch

`--overlap` publishes reads and resumes the graph instead of waiting for the complete batch.
The fork's `ggml_cpu_set_expert_ready_hook` runs before a CPU expert kernel consumes a slice.
Readiness is tracked per **`(projection, expert)` plus generation**, for the layer in flight.
A hit is marked ready immediately; a completed read publishes its flag. An unready consumer
briefly spins, then may wait on a condition variable. This does not require all of an expert's
gate/up/down weights to arrive before its first projection can run.

The async path orders demand jobs projection-major and experts in the kernel's visitation
order. It is not arbitrary out-of-order execution of whichever expert happens to finish first.
It drains the prior batch before reusing flags/jobs. The default path does not predict the next
layer's routing: temporal/predictive prefetch and route-ahead are separate features.

## Defaults and model compatibility

| Setting | Current host CLI behavior |
|---|---|
| `--moe-stream` | Off unless requested; omission is the ordinary mmap baseline. |
| Expert cache with streaming | CLI defaults to `auto`; the library config itself defaults to cache off. |
| `--cache-floor-mb`, `--cache-ceil-mb` | 1536 MiB headroom; ceiling 0 means no extra user ceiling. |
| Numeric cache budgets | 0 disables caching; values below the 1500 MiB validation floor require `--force-cache`. The model's token-cycle working set is a separate quantity. |
| `--io-threads` | 4 by default, at most 8. Not the compute-thread count. |
| Expert direct I/O | Requested by default; check effective `o_direct`, not just flags. |
| `--dense-weights` | `anon` by default when streaming is active. |
| `--overlap` | Off in the CLI; requires the compiled-in ready hook. |
| `--row-stream` | Off; optional row window defaults to 64 MiB. |
| `--release-mmap` | Off; release is conditional on rebinding/safety checks. |
| Sampling / context / compute threads | Greedy `temp=0`; context 2048; compute threads 4; `--ubatch 0` follows the batch width. These are defaults, not benchmark recommendations. |

The registered architectures at the investigation checkpoint are `qwen3moe`, `qwen2moe`,
`qwen35moe`, `gemma4`, `gpt-oss`, `lfm2moe`, `deepseek2`, `deepseek4`, `bailingmoe3` and
`qwen4exp`. `--list-archs` is authoritative. Most recipes name split gate/up/down tensors;
Gemma 4 uses fused gate+up. Unregistered/dense models may still run through the mmap baseline,
but cannot be assumed to support `--moe-stream`. Split GGUF sets are opened from the first shard.

DeepSeek-V2-Lite uses `deepseek2`: one leading dense layer, shared experts and MLA attention
stay on the dense policy; its routed tensors use the standard split layout. These reference
model facts must never become constants in the streaming implementation.

MTP and n-gram speculative decoding exist. Prefill and wider verification batches stream a
union of routed experts; that can reduce sparsity without making batching categorically
unsupported. `--mtp` needs an exported trained head; `--ngram` does not.

Losslessness claims require the lossy knobs to be off: expert-count overrides, cold-expert
dropping, cache-aware substitution and route-ahead can change output. Cache-dependent dropping
and substitution can also make repeated output non-reproducible. Stochastic sampling has its
own seed/temperature contract. Do not transfer Android app defaults to the CLI by assumption.

## Page cache: exact scope of bypass

Linux expert misses normally use `O_DIRECT` plus positioned reads. Reads are aligned into a
per-lane bounce buffer and copied to the canonical weight slice: **not zero-copy and not
kernel bypass**. FileReader verifies an aligned block against a buffered read at open.
Unsupported direct opens or failed verification can downgrade the file to buffered I/O;
sub-alignment EOF tails also use a buffered descriptor.

| Weight policy | Residency and file-cache behavior |
|---|---|
| `mmap` | File-backed pages, served through normal faults/page cache. |
| `warm` | Same mapping, deliberately populated through a buffered sweep. |
| `anon` | Independently requested direct reads into anonymous buffers, then tensor rebinding. Anonymous memory can be swapped where swap is allowed. |
| `ahwb` | Android-only reclaim-exempt allocation with the same loading/rebinding idea; unsupported platforms fail rather than silently becoming `anon`. |
| Row streaming | Opt-in direct readers and a bounded row window for tables the captured graph can safely serve by row. |

`--no-odirect` changes **expert readers only**; it does not disable the dense-`anon` or row
reader's independent direct request. The run's `o_direct` describes the expert shard readers'
effective mode, not proof that every file access in the process bypasses caching.

Initialization still uses llama.cpp's file mapping and a capture decode. The Linux smoke trace
observed `MAP_SHARED|MAP_POPULATE` and `MADV_WILLNEED` before later successful direct reads.
Do not generalize that exact prefetch pattern to every architecture or platform. Oversized or
unrebound tensors can remain mapped. `MADV_DONTNEED` on a mapping is not a global file-cache
flush, and later `--release-mmap` cannot undo earlier cache population.

## Memory budgets and cgroups

`auto` is a **one-time** budget calculation, not a pressure controller. It reads platform
available memory, reserves the pending dense anonymous/pinned conversion and any row window
as applicable, subtracts configured headroom, and clamps to the model's expert-bank size and
optional ceiling. The exact implementation is `ExpertStreamSource::init`.

On Linux, `platform_io::mem_available_bytes()` reads host `/proc/meminfo` `MemAvailable`.
It does **not** currently incorporate cgroup `memory.max`/`memory.current`. A large host can
therefore produce an overlarge cache budget inside a small cgroup. The kernel still enforces
the cap; with swap disabled, an unaffordable anonymous working set can be OOM-killed rather
than causing automatic LRU resizing. Do not silently change cache policy between experimental
arms to conceal this outcome. See [cache sizing](docs/cache-sizing.md).

For controlled cgroup-v2 measurements:

- Use a fresh group/process per cap; user-level `systemd-run` works only when the user manager
  has delegated the memory controller. Do not assume root privileges are required or available.
- Verify the **actual model PID's** cgroup and kernel `memory.max`/`memory.swap.max`, not merely
  the launcher arguments. Record CPU affinity, cache budget, dense mode and effective direct I/O.
- File pages may remain charged to the group that first populated them. For an experiment-owned
  model, flush pending writes and use targeted `POSIX_FADV_DONTNEED` before fresh-group loading,
  then inspect `memory.stat:file`. Advice is best-effort; **never drop the global page cache**.
- Record `memory.current`, anon/file statistics, `memory.events`, pressure and swap usage at
  explicit stages. This host did not expose `memory.peak`; a snapshot or sampled maximum must
  not be presented as an exact peak.
- Preserve `Result=oom-kill`/exit evidence before cleanup. An OOM during warm-up has **no valid
  hot-state breakdown**, not zero-cost I/O or a successful smaller workload.
- Supervise the real transient unit. Stop that unit after completion and clean failed units;
  terminating only the `systemd-run` wrapper need not terminate the model process.
- An output-pattern wait can outlive a process that already died. Check terminal process/unit
  state as well as `BMOE_DONE`; do not wait indefinitely for an impossible marker.

The investigation used `memory.max` of 4/8/16/32 **GiB** with `memory.swap.max=0`, not the
same numbers as a `--cache-mb` budget. CPU-only runs used 32 compute threads and an affinity
mask verified to cover 32 physical cores on one NUMA node. `taskset` is **not exclusivity**:
other processes can still compete on those cores. This was an execution-policy experiment,
not an ARM/UFS, thermal-throttling or phone-performance simulation.

## Models and conversion

Use the pinned official converter and native quantizer. Select external model/output paths
with the user; do not modify original checkpoints or commit the generated weights.

```bash
CUDA_VISIBLE_DEVICES= python3 third_party/llama.cpp/convert_hf_to_gguf.py \
  "$MODEL_SOURCE" --outfile "$GGUF_DIR/DeepSeek-V2-Lite-F16.gguf" --outtype f16
CUDA_VISIBLE_DEVICES= build/bin/llama-quantize \
  "$GGUF_DIR/DeepSeek-V2-Lite-F16.gguf" \
  "$GGUF_DIR/DeepSeek-V2-Lite-Q4_K_M.gguf" Q4_K_M 32
```

The actual DeepSeek conversion produced 404 tensors, including 78 routed-expert tensors.
F16 was 31,424,035,840 bytes (29.266 GiB); native Q4_K_M was 10,367,958,016 bytes
(9.656 GiB). The latter is mixed quantization: 54 tensors required native fallback; routed
weights include Q4_K, Q5_0 and Q8_0. Do not describe it as uniformly four-bit. Both exports
were retained in the user-selected directory. When present locally, resolve their exact
paths from `build/measured-deepseek-cgroup/conversion.json`; do not publish host paths.

`scripts/make-tiny-moe.py` produces deterministic **random F32** Qwen3MoE/Gemma4 fixtures,
not language-quality models. Its default fixture is small enough to test routing and identity,
not realistic bandwidth or >RAM behavior. The larger F32 experiment used in-memory shape
overrides to that generator; there is no invented `--scale` CLI flag.

## Traces and measurement semantics

| Instrument | What it establishes |
|---|---|
| `--csv` / `BMOE_*` | Per-token/run telemetry. `compute_ms` is a residual, not an independent kernel CPU clock. |
| `--route-trace` | Selected expert IDs, weights, cache residency and attributed bytes, keyed by turn/phase/step/layer/slot. |
| `--compute-trace` | v2 monotonic graph-node, callback and enclosing decode-frame intervals; expensive per-node isolation. |
| `--compute-trace-layers` | Coarser segments with fewer barriers; includes callback/load work, not isolated kernels. |
| `--io-trace` | v2 per-reader-service intervals and per-worker unmet-ready waits, including their start/end timestamps. |

v2 compute and I/O files share `clock=steady_ns` and a session `trace_id`. Pair only matching
IDs. `BMOE_DECODE` is an enclosing frame, not another operation to add to node totals.
`BMOE_CALLBACK` separates callback/cache work from the following node. `MUL_MAT_ID` covers a
whole expert-matmul graph node, **not an individual expert's compute duration**.

Do not conflate these quantities:

- Graph wall time includes synchronization, scheduling and readiness waits. Process `cpu_ms`
  includes **all** threads, including I/O work and spinning; dividing by compute threads does
  not isolate arithmetic CPU time.
- In normal serial telemetry, `io_ms` measures the blocked read phase. Under overlap it uses
  summed reader syscall busy time; that is not a disjoint part of decode wall time.
- Cache-hit percentages, eviction counts and re-read counts are cumulative source counters.
  They can include prefill and earlier session turns. A hot turn can issue zero reads while
  its displayed cumulative hit percentage remains below 100%; use turn-filtered events/bytes
  or explicit counter deltas to describe that turn.
- Trace `kind=read` brackets `FileReader::read`, including alignment/copy work and scheduling,
  not just storage-device service. `kind=wait` is one worker's unmet dependency.
- Union concurrent intervals before wall-time attribution. Any-worker wait union is not proof
  that all workers were idle. Overlap means percentages need not add to 100%.
- Zero explicit expert read bytes/events cannot be explained by CPU load "hiding" those reads.
  It does not imply zero system-wide disk traffic: logging, mappings and other processes differ.
- `o_direct=1`, cache-hit percentage and time residuals alone do not prove every access is uncached
  or establish a compute/storage bottleneck. Tracing itself changes scheduling.

```bash
python3 scripts/decode-analyze.py timeline compute.csv --io io.csv \
  --turn 1 --svg timeline.svg --summary timeline.json
python3 scripts/route-analyze.py route.csv --view overlap --phase decode
python3 scripts/route-analyze.py route.csv --view workset --phase decode
```

Use explicit `--turn`/`--phase` when excluding warm-up. `--step` selects a frame and `--layer`
zooms the plot; see [telemetry](docs/telemetry.md) for exact aggregation scope. Duration-only v1
traces cannot reconstruct chronology. Standalone I/O summaries see **read-active frames**,
not cache-hit-only frames; use paired compute frames for a full-run denominator.
The route analyzer's `--view overlap` measures agreement of expert selections, not the
runtime's `--overlap` scheduling feature. For ordinary one-token decode, CSV progress step 1
corresponds to trace position `n_prompt`; account for the one-based offset and the session turn
when joining them. Speculative batches and reused KV need their actual positions, not a
blind prompt-length subtraction.

## What the investigation established

These are **historical measured configurations**, not new test results for every later edit.
The code checkpoint was `9f9315e`; it reported engine 0.23.0 while the changelog already
started at 0.24.0. The dependency was `0e8c83e512b99ccf83e50798b86f0b0ec40a7b0a`
with the optional hook. Do not silently align version labels or claim a new release.

| Experiment | Established result and boundary |
|---|---|
| Tiny random Qwen3MoE/Gemma4 GGUFs | CPU build and CLI execution worked; streamed/resident outputs matched for the exercised 32-token cases, including buffered/direct reads and selected cache/overlap variants. Not a quality or large-model benchmark. |
| About 818 MiB F32 model; 768 MiB expert bank, cache artificially capped at 128 MiB | Serial traced mean 49.73 ms/token: reader-service union 67.12%, graph elapsed 23.92%. Overlap existed but was not faster: untraced medians 23.707 vs 17.807 tok/s (serial vs overlap, three runs). This was a **cache-pressure** case, not unlimited-memory hot decode. |
| Same F32 model, full-capacity cache and same-process warm replay | Discarding a complete warm-up then replaying three identical 32-token requests gave 96 hot tokens with zero expert reads; traced mean 5.783 ms/token. Clearing KV did not clear expert weights. |
| F32 model under 4/8/16/32 GiB cgroups, both run orders | All caps exceeded the approximately 1.60 GiB charged usage observed after measurement. Each cap had 192 hot tokens, zero expert reads and no memory-pressure/OOM events. Timing variation, including one long-tail 8 GiB run, was not evidence of memory-limit degradation. |
| Real DeepSeek-V2-Lite Q4_K_M, full-capacity `auto` policy | Resident and streamed CPU output matched for 32 tokens. At 4 GiB, warm-up was OOM-killed. At 8/16/32 GiB, each run completed 96 hot replay tokens with zero expert reads/major faults and traced means 31.580/31.412/31.381 ms/token. |

For the real model, `auto` selected 9086 MiB, but this repeated request populated only
6859 MiB of expert weights (about 75.5% of bank **bytes**, not a measured expert-ID fraction).
Anonymous memory was about 7.53 GiB. End-of-measurement group usage was approximately
8.00/16.00/17.30 GiB at the successful caps, with differing amounts of reclaimable file cache.
The 8/16 GiB cases did hit their memory caps; this did not force rereads of already-resident
anonymous expert slices. It does not mean every expert or every longer workload fits in 8 GiB.

The complete breakdown tables, model details and portable recipes are in
[CPU profiling findings](docs/cpu-profiling-findings.md). Local evidence, when retained:

- `build/smoke/`: tiny fixtures, successful direct-I/O syscalls and initial diagrams.
- `build/measured-cpu32/`: scaled fixture and cache-pressure/overlap controls.
- `build/measured-cpu32-warm/`: corrected same-process hot replay.
- `build/measured-cgroup-memory/`: synthetic cap sweep and reverse-order repeats.
- `build/measured-deepseek-cgroup/`: conversion metadata, real-model checks, cap outcomes and CSVs.

These ignored directories are not part of a clean clone. Never claim new measurements when
only rereading them. The original `expert-cache-gantt` illustration in `build/smoke/` was
**schematic**, with arbitrary units; later v2 timelines were generated from real timestamps.
Do not mix the two as evidence.

## Interpretation limits and deferred work

### Changing experts is not the same as missing the cache

For one layer, let `S[t]` be this token's selected experts and `C[t]` its resident experts.
Adjacent-token change concerns `S[t]` versus `S[t-1]`; misses concern `S[t] - C[t]`.
Completely different consecutive selections can both be cache hits. The same numeric expert
ID in another layer is another cache entry.

Repeating an identical prompt and greedy continuation with cleared KV reproduces the routing
sequence. It tests a warmed **known trajectory**, not sustained generation of novel tokens.
The 32-token studies did not measure adjacent-token turnover, long-text coverage growth or
diverse-prompt generalization. Zero replay reads is not evidence that experts rarely change.

The user explicitly paused new tests to discuss this and possible CPU contention. Do not start
the deferred study just because it appears here. When explicitly resumed, distinguish:

- A fully resident compute baseline from genuinely memory-constrained streaming.
- One excluded startup/warm-up period from new expert first-touches during later generation;
  those later misses are real streaming costs, not samples to delete after the fact.
- Continuous longer generation and diverse prompts from repeated 32-token replay. Report actual
  token counts if generation stops early and inspect degenerate repetition.
- Per-layer route overlap, cumulative unique `(layer, expert)` coverage, byte-weighted working
  sets, first-touch misses, eviction rereads and measured read traffic.
- CPU execution from scheduler contention/spinning. Affinity does not reserve cores. CPU load
  was raised as a possible confound, but the responsible processes and their causal effect
  were **not** established.
- Changes in expert working set from the extra KV/graph memory caused by a larger context or
  ubatch. Hold these settings equal across comparison arms and report both traced and untraced
  controls where making a throughput claim.

### Known unresolved trace boundary

Static review identified a cancellation/error edge case in `Session::generate`:
`begin_compute_batch()` appends a `BMOE_DECODE` frame with `end_ns=0`, but some prefill/decode
failure or cancel exits bypass `trace_flush()`. Reusing the session can then flush an old
unfinished frame, which the timeline parser correctly rejects. This limitation is recorded
in the checkpoint commit; **it has not been fixed or covered by a cancel-then-reuse trace
regression**. Close or discard frames on every exit and add that regression when addressing it.
Do not advertise ordinary successful multi-turn runs as proof of cancellation correctness.

The completed validation was a successful host build, 13/13 existing CTest entries, format
checks, real-model streamed/resident byte comparisons, measured timestamp/frame/union checks,
rejection of mismatched trace pairs and a real all-cache-hit timeline. Documentation-only work
should check links, source claims and arithmetic without silently restarting paused benchmarks.
