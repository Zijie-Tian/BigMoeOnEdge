# Limitations and prior art

## Prior art

BigMoeOnEdge is an engineering package, not a new technique. The ideas it combines:

- **AirLLM** — layer-by-layer streaming of >RAM models from disk.
- **Apple, "LLM in a flash"** — flash-aware weight streaming, windowing, sparsity-driven
  loading.
- **FlexGen** — offloading and I/O-bound throughput scheduling for large models.
- **PowerInfer / EdgeMoE** — hot/cold expert locality and expert-granularity residency on
  the edge.

The contribution here is a clean, modular, llama.cpp-native implementation of
expert-selective streaming that stays lossless and runs on the public API — no fork for the
serial path, and only a single ~25-line hook (with an explicit sunset) for the optional
`--overlap` feature. See [seam.md § 3](seam.md).

## Limitations

- **Reproducibility depends on routing and sampling.** `--n-expert-used` deliberately changes
  routing width and output. [`--drop-cold-experts`](expert-dropping.md) and
  [`--expert-substitute`](cache-aware-substitution.md) additionally decide from live cache state,
  so even with greedy sampling the same prompt and flags can decode differently across runs.
  The gates cover their machinery, not a promise of byte-identical output. Both are off by
  default in the CLI; use `--ppl --ppl-step` rather than wide scoring batches to price them.
  When stochastic sampling is enabled, also record temperature and seed rather than assuming
  every non-cache setting preserves deterministic output.
- **Sparsity shrinks with the routing union.** Single-token greedy decode is the sparse case
  (top-k experts per layer). Prefill is batched (`n_batch`); `--mtp` and `--ngram` speculative
  decode exist (`core/src/engine/session.cpp`, `core/src/engine/ngram_draft.cpp`) and verify a
  wider batch. Streaming is not categorically incompatible with batching or speculation, but
  the load is the union of the batch's routed experts — larger than one token's set, and able
  to approach the full bank as the batch grows.
- **CPU-only today.** `Session::open` sets `n_gpu_layers = 0` for every layer, not merely the
  streamed experts; the rebind targets host memory. GPU offload is not a current path. Wall
  time is not always flash-I/O-bound: CPU contention can dominate or distort timing
  attribution even while expert read bytes and events remain real.
- **Shared experts stay resident.** Architectures with an always-on shared expert (e.g.
  `gemma4`, `deepseek2`, `deepseek4`) stream routed experts but keep shared experts and dense layers
  resident (in the page cache, or in the engine's own buffers under `--dense-weights anon`),
  so the streamed fraction (and the memory saving) is smaller than for a purely routed model
  like `qwen3moe`. The same applies to architectures whose first blocks are dense by design
  (`lfm2moe` has a `leading_dense_block_count`): those blocks name no expert tensors, so they
  are never streamed.
- **A resident tensor can be larger than RAM, and then it is only ever mmap'd.** `qwen4exp`
  (Qwen3.8-Flash-Next) carries a 51B n-gram embedding table (`per_layer_token_embd`, ~28.8 GB at
  IQ4_NL) that the graph reads sixteen rows at a time through `get_rows`. It is not indexed by
  expert, so the streamer does not bind it, and it is bigger than any phone's memory, so no dense
  policy can make it resident: such a tensor stays mmap'd whatever `--dense-weights` asks, and
  the engine says so at load. The streamed fraction of this architecture is therefore unusually
  low, and its per-token cost on that table is page faults on kilobyte reads rather than
  streamed expert bytes. That cost has not been measured on a device yet.
- **Streaming does not help a model that fits.** The engine's reason to exist is a model
  larger than RAM. Registering an architecture says the layout streams losslessly, not that
  streaming is the fast way to run every model using it — a small MoE that fits in memory is
  faster loaded resident, and the registry rows are about coverage, not a recommendation.
- **Repack must stay off.** Loading uses `use_extra_bufts=false`; you cannot combine
  streaming with weight repacking.
- **macOS reads uncached, and `o_direct` says so — but it is not `O_DIRECT`.** A direct request on
  Apple is served by `fcntl(F_NOCACHE)` on each descriptor: the kernel stops caching that file's
  pages, which is the property the design wants. It is a caching hint, not an I/O mode — no
  alignment contract, no DMA promise — so reads keep ordinary `pread` semantics and the raw-read
  ceiling can sit below Linux `O_DIRECT`'s. Measured on one 16 GB Apple-silicon Mac with the model
  on an external volume (256-token protocol, buffered vs `F_NOCACHE`, interleaved A B B A A B):
  decode 0.94 vs 0.61 tok/s (+56 %, arm ranges non-overlapping), with the byte stream
  bit-identical between arms (40 822.8 MiB read by both) and cache-hit equal — the gain is not
  fewer reads but the buffered arm's page-cache pollution doubling the compute residual
  (1.22 → 0.60 s/tok) while stall stays flat (0.43 → 0.46 s/tok). The `o_direct` field records
  the open's real outcome on every platform, so a refused or downgraded run reports `0`.
- **No iOS target.** The core is portable C++ and the streaming path has no Android dependency, but
  there is no Xcode project here and iOS does not run command-line binaries, so there is no
  supported way to run or benchmark the engine on an iPhone or iPad.
- **Windows throughput.** The cache's reserve-then-commit-per-slice path is heavier on
  Windows than the POSIX lazy-commit path. The gates run on Windows; the throughput
  targets are stated for Android/Linux.
- **Depends on a ggml scheduling behaviour** (documented in [seam.md](seam.md)) that is
  not a stability-guaranteed contract. Re-verified by the gates on each submodule bump.
- **Compute-trace frames can leak across a failed or cancelled generate.** `RouterHook::begin_compute_batch`
  (`core/src/moe/router_hook.cpp`) opens a `BMOE_DECODE` row with `end_ns = 0`. `Session::generate`
  (`core/src/engine/session.cpp`) only runs `trace_flush` → `end_compute_batch` after a successful
  `llama_decode`. Prefill and decode error/cancel exits can return or break without flushing, so
  the open frame stays in `compute_rows_`. A later successful generation may flush that invalid
  row with the new one; a timeline that requires a closed interval will reject it. Static reading
  of those functions, not reproduced by a cancel regression. Normal, non-cancelled runs were
  validated. This is not a claim that the code has been fixed.
- **Zero expert I/O on a hot replay is cache, not routing stability.** Repeating an identical
  short greedy request with `clear_kv = true` retains expert weights and replays routing; zero
  expert reads is not evidence that adjacent tokens or long/new text would also need no I/O.
  CPU contention can distort wall-time attribution but cannot turn actual read bytes/events
  into zero. `taskset` is affinity, not CPU exclusivity. Per-node graph wall includes sync,
  scheduling and waits (`core/include/bmoe/decode_trace.h`); `FileReader` intervals include
  copies; summing those intervals is a union, not a per-lane or per-thread duration.

## Not goals

- Distributing a model across devices (a different axis).
- Beating a model that already fits in RAM — if it fits, run it resident.
