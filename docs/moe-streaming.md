# MoE expert-selective streaming

## The lever

A MoE layer stores `n_expert` experts (128 for Qwen3-30B-A3B) but each token is routed to
only its top-`k` (8). The other experts' weights are never read for that token. If the
model does not fit in RAM, streaming just the routed experts from flash turns "the whole
expert bank per token" into "top-k/n_expert of it" — about 6% for that model.

This sparsity is largest for **autoregressive, one-token-at-a-time decoding**. A batch of `T`
tokens routes the *union* of those tokens' top-k sets, which grows toward the full bank as `T`
grows — wider routing unions reduce sparsity. Prefill is already batched (`n_batch`; one-shot
`run()` uses `n_batch = n_ctx`). Optional `--mtp` and `--ngram` speculative decode exist and
verify a wider batch; streaming is compatible with both, but the expert I/O is that union, not
a single token's top-k. Default greedy decode still submits one new token per graph unless
speculation widens it.

## Mechanism

1. **Bind.** After capture warm-up (see [seam.md](seam.md)), each bound MoE layer's expert
   tensors are rebound onto streaming buffers: split gate/up/down or a recipe's fused gate+up
   layout. Dense layers are not bound as routed experts.
2. **Route.** Normal demand loading observes `ffn_moe_topk-<il>`; optional diagnostics and
   routing policies may request additional nodes. ggml computes and synchronizes the needed node.
   The selected IDs are gathered **respecting the view strides** — `selected_experts` is a view of the full
   argsort with row stride `nb[1]`, so a flat read would grab the wrong experts and corrupt
   the KV cache.
3. **Load.** The expert source reads those experts' slices from the gguf into each expert's
   canonical offset inside the bound tensor, just before that layer's expert matmul runs.
   Linux misses request `O_DIRECT` (`FileReader` in `core/src/io/file_reader.cpp`), with a
   buffered verify, a buffered fallback if the verify fails, and a buffered sub-alignment EOF
   tail. `--no-odirect` only changes expert readers; dense anon/pinned uses its own
   `FileReader`, independent of that flag. Direct expert I/O does not eliminate the initial
   mmap or capture-warmup page cache (`LLAMA_LOAD_MODE_MMAP` in `Session::open`).

Ordering is guaranteed by ggml's eval-callback loop: the node we mark is computed and
`ggml_backend_synchronize`'d before the non-ask callback fires, and the following compute
(the expert matmul) runs only after our load returns. The next layer cannot overwrite the
buffers until this layer's matmul has synchronized. This describes serial streaming on the
current CPU path (`n_gpu_layers = 0`), not correctness on arbitrary ggml backends. With
`--overlap`, load submission returns before all reads finish; the fork's expert-ready hook
then gates each projection/expert before consumption, as described in [seam.md](seam.md).

The ordinary streaming path is **lossless** with unchanged routing and sampling, asserted by
the byte-identity gates. Expert-count overrides, [cold-expert dropping](expert-dropping.md),
[cache-aware substitution](cache-aware-substitution.md) and [route-ahead](route-ahead.md) are
separate, opt-in policies that deliberately change routing/output. Do not transfer the
losslessness claim to those settings; they are off in the default greedy CLI configuration.

## Residency modes

- **Cache off (shared slots).** One full-size heap slot per present projection is shared
  across layers — one layer computes at a time. Routed slices are reread each step. This avoids
  cross-layer expert residency, at the cost of repeat I/O.
- **LRU cache (`--cache-mb N`).** Each `(layer, projection)` gets a reserved,
  lazily-committed address range. Cache *keys* are `(layer, expert)` (`id = il * n_expert + e`
  in `ExpertStreamSource`). A routed expert already resident is a **hit** (no read);
  a miss is read once and kept; over budget, the coldest `(layer, expert)` is evicted and
  its pages physically released (`madvise(MADV_DONTNEED)` / `MEM_DECOMMIT`). The budget targets
  expert residency, not total process memory; current/in-flight entries remain protected.
  Overlap readiness is separate: one cell per `(projection, expert)`, valid when
  `gen == async_gen_` for the in-flight layer (`ReadyFlag` in `core/src/moe/expert_stream_source.h`).

### The cache rule: 0 or ≥ ~2 GB

Expert reuse is broad, not skewed: hit rate rises roughly linearly with budget, with no
small-cache plateau. A budget below one token's routed working set (~1 GB for
Qwen3-30B-A3B) yields zero hits **and** pays eviction overhead — measurably slower than no
cache. So `validate()` rejects a budget in the `1..1499 MiB` band unless you force it. Use
`0`, or `≥ 2000`.

## Parallel reads (`--io-threads N`)

Routed slices are read across `N` lanes, each with a private fd and bounce buffer; the
calling thread participates as lane 0. On UFS 4.x, 4 lanes roughly triples effective read
bandwidth over serial. Compute threads (`-t`) show a U-shape — 4 is the measured optimum;
8 regresses badly because ggml's spin-wait contends with the synchronous reads.

## The model file's mapping (`--release-mmap`)

llama.cpp maps the gguf and keeps it mapped for the model's lifetime. On Windows that mapping
serialises the lanes above: while a section of the file is alive, concurrent unbuffered reads on
it are taken one at a time, so `--io-threads 4` reads at one lane's rate. It is not the drive and
it is not the engine — `bmoe-iobench --model M.gguf --lanes 4 --slice-kb 576` measures 2400-2660
MiB/s, the same command with `--mmap` measures 895-930, and adding `--reopen-lanes` recovers the
full rate. A lane opened while the section existed stays serialised after it is gone, which is why
the recovery needs both halves.

`--release-mmap` does exactly that inside the engine: after load, once nothing reads through the
mapping any more, the file is unmapped, its section closed and the reader lanes reopened.

Whether releasing is safe is decided by looking, not by reasoning about which tensors ought to have
been rebound: the engine asks the OS whether any weight the capture pass observed still points inside
a mapping of the model files, and declines if any does. That one question covers every residency
policy — a dense set left mmap'd under `mmap` or `warm`, a table held back as oversized, or a tensor
no name-based accounting could have found. Run with `--dense-weights mmap` and the engine reports the
count and stands down.

On Windows the run ends with `warning: UnmapViewOfFile failed`, printed by llama.cpp rather than
by the engine. That is the designed outcome, not a defect: llama.cpp is not patched and still
believes it owns the mapping, so at teardown it unmaps a base the engine has already released. An
inert reservation is left in that range precisely so the call finds a placeholder and fails
harmlessly, instead of finding whatever was allocated there next.

It is still opt-in, because the check answers for the pointers the capture pass saw and for no
others. A graph shape this session never builds could hold another one, and llama.cpp exposes no way
to enumerate a loaded model's tensors and settle it. The MTP draft is the concrete case: it builds a
second graph, so a gguf tensor no policy owns blocks the release there as well. Worth +46% decode on
the desktop host, byte-identical.

Android is a different story with the same conclusion. The iobench cells above are flat there — f2fs
does not serialise, so there is no read bandwidth to recover, and the engine's flash stall is
unchanged with the flag and without. What changes is CPU: a 20 GB mapping the kernel still has to
account for costs about 9% of the decode's CPU time on a device under memory pressure, and dropping
it is worth 5-9% of throughput. That measurement is two short cells per variant and is a direction,
not a number. The flag stays off by default on both platforms.

## Why repack must stay off

The streamer rebinds `tensor->data` to a buffer it fills from the file's native byte
layout. `use_extra_bufts=true` would repack Q4_K weights into a different in-memory layout
(e.g. `q4_K_8x8`), so the file offsets would no longer describe what the matmul reads.
The engine loads with `use_extra_bufts=false`; this is load-bearing, not a tuning knob.

## Assumptions to re-check on a submodule bump

- The routing node is named `ffn_moe_topk-<il>` and the expert tensors
  `blk.<il>.ffn_{gate,up,down}_exps.weight`. The recipe isolates these names.
- The eval-callback fires per decode (not skipped by graph reuse) and computes a
  marked node alone before the non-ask callback. The gates catch a regression here.
