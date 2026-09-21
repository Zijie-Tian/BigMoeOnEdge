# Expert cache sizing

The expert cache budget (`--cache-mb`) is the dominant throughput lever, but the right value is
device- and model-specific: too small and the hit rate collapses **to exactly zero — see the
cliff below, it does not degrade gracefully**; too large and the pinned cache
plus the mmap-resident model push `MemAvailable` to zero and the Android low-memory-killer takes
the process (this is exactly why Gemma cannot use a 4000 MiB cache on an 11 GB phone — see
[benchmarks.md](benchmarks.md)). `--cache-mb auto` removes the guess: the engine sizes the cache to
the device once at load and holds that budget for the whole run.

## The cliff: below one token cycle the hit rate is 0 %, not "low"

Global LRU drops the least *recently* used entry. The model visits its layers in a fixed cycle
0..N-1, so the least recently used entry is also the **soonest to be needed again**. While the
budget holds a whole token cycle the cold end is genuinely stale history and this never shows.
Once it does not, the cache evicts precisely what it is about to read, and the hit rate does not
taper — it goes to **exactly 0.0 %**.

Measured on device (2026-07-20): gpt-oss-120b at top-4 has a `token_demand_MiB` of 1815, and a
**1500 MiB** budget returns 0.0 % hit while reading 1817.05 MiB/token — *identical to running with
no cache at all* (1817.0) — and still spends 0.233 s/token, 24 % of the token, managing it, on top
of holding 1500 MiB of RAM. Dropping the budget 2000 → 1500 does not cost a quarter of the hit
rate; it costs all of it.

Two traps worth knowing:

- **1500 is a legal value.** It is exactly `cache_min_mb`; no `--force-cache` is needed. Today's
  only protection is that this floor happens to sit above the cycle for the shipped models at
  their default top-k. Raise `--n-expert-used` and the cycle moves above the floor (gpt-oss: 908
  MiB at k=2, 1815 at k=4) and the protection silently stops holding.
- **The existing floor is a different quantity.** `layer_demand_MiB` (the widest single layer, 50.4
  on that run) is the mechanical minimum; the cliff is at `token_demand_MiB` (1815). They are
  roughly `n_layer` apart. Both are already printed in the `# summary` line; nothing sizes from them.

So when a budget is being chosen by hand, check it against `token_demand_MiB` from a previous run
of the same model *and top-k*, not against `cache_min_mb`. Below the cycle, `--cache-mb 0` is
strictly better than a cache: same reads, none of the RAM, none of the management cost.

Fixing this by changing the eviction policy was tried and rejected — a per-layer partition is
immune to the cliff by construction but costs ~30 % throughput
([bench-data/2026-07-20-cache-replay/layer-lfu-verdict.md](bench-data/2026-07-20-cache-replay/layer-lfu-verdict.md)).
The cheap fix was a guard, and it is in: the worst-case cycle is priced at init from the model's
shape alone — every bound layer's expert-entry bytes times `min(top_k, n_expert)`, at the top-k the
run actually applies. The engine records it in every metrics preamble as `cache_cycle_mb` next to
the budget it is being compared against, and prints one line to stderr at load when the budget is
under it:

```
bmoe: WARNING expert cache 1500 MiB is below this model's worst-case token cycle of 1815 MiB
at top-k 4 — no entry can survive to the next token, so expect a hit rate near zero.
```

It states the fact and stops there; the advice above — that `--cache-mb 0` is strictly better below
the cliff — stays here rather than in the engine's output. Note this is the *worst case* (a disjoint
routed set at every layer), so it is an upper bound on the measured `token_demand_MiB`, and a budget
above it is safe for any routing. Recording it matters as much as the warning: a committed CSV whose
budget sat under the cycle now says so, without needing a run of the same model to compare against.

## What it does

- **At init**, once the full expert-set size is known, the budget is set to
  `available_RAM − cache_floor_mb`, clamped to `[cache_min_mb, total expert bytes]`. Available memory
  is read from the platform (`/proc/meminfo` `MemAvailable` on Linux/Android, `GlobalMemoryStatusEx`
  on Windows); if it is unknown the budget falls back to the `cache_min_mb` floor.
  Under `--dense-weights anon` or `--dense-weights ahwb` (the internal `Pinned` policy),
  `ExpertStreamSource::init` subtracts pending dense allocations and any row-gather window
  first, skipping tensors that will stay mmap'd because they exceed available RAM.
  `--dense-weights mmap` / `warm` do not convert those bytes, so they are not deducted.
  The Linux reading is host `MemAvailable`; it does not honour cgroup `memory.max`.
- **Also at init**, the dense (non-expert) regions of the gguf — header, embeddings, attention,
  norms, lm_head, the tensors the streamer leaves mmap-resident — are warmed into the page cache with
  one sequential buffered sweep (reported as `bmoe: dense warm-up`), so the first tokens do not pay
  for them as random 4 KiB faults. On a model far larger than RAM this is the difference between a
  fast first token and a ~20-token slow-start ramp: measured on gpt-oss-120b, the first-five-token
  wall average drops ~20× (see [benchmarks.md](benchmarks.md)). On models whose dense set is small it
  is a harmless no-op. This sweep is the `--dense-weights warm` policy; `--dense-weights mmap`
  disables it for A/B runs, and `--dense-weights anon` (the default) replaces it with an O_DIRECT
  read into anonymous buffers, which is the better answer once the model is well past RAM — the
  case the engine targets.
  The warm-up is deliberately kept *out* of the budget: it only pre-faults the mmap-resident pages,
  it does not pin or reserve them, so the expert-cache budget above is unchanged and its hit rate is
  identical with and without it. (An alternative that folds the dense bytes into the floor —
  reserving RAM so the expert cache can never evict them — was measured and rejected: on a
  cache-sensitive model it lowers the budget and the hit rate, e.g. Gemma budget 4000→2909 MiB, hit
  83%→73%, trading throughput for OOM headroom that the warm-up already avoids needing. See
  [bench-data/2026-07-14-warmup/](bench-data/2026-07-14-warmup/).)
- **During generation, nothing resizes it.** `auto` is one shot at load, not a control loop: the
  budget chosen at init is held for the whole run. A runtime governor that tracked free RAM and
  shrank the budget under pressure did exist and was **retired** — it was measured a net loss on
  the models it was built for (see [pressure.md](pressure.md)). The `moe-cache:` summary reports the
  budget and what actually stayed resident:

  ```
  moe-cache: 77.1% hit, resident 4000.0 MiB
  ```

Expert miss reads request cache-bypassing I/O (`O_DIRECT` on Linux when the open and the
verify succeed). Shrinking the expert cache is what hands those committed expert pages back.
That is not the engine's only large allocation, and it is not a claim that load never
touches the page cache: `Session::open` still mmaps the gguf (`LLAMA_LOAD_MODE_MMAP`), and
`FileReader` can fall back to buffered I/O or a buffered EOF tail. Under `--dense-weights anon`
or `ahwb` (the internal `Pinned` policy; anon is the default), the dense set is converted into engine buffers; that
footprint is not part of the expert LRU, but `ExpertStreamSource::init` *does* deduct those
pending allocations from the `MemAvailable` reading before `--cache-mb auto` chooses a
budget. `--dense-weights mmap` / `warm` leave dense weights file-backed, so that deduction
does not apply — `MemAvailable` still counts those mmap pages as free. See below.

> **The budget is not only a throughput knob — it is what the kernel judges you by.** On Android the
> LRU promotes a page to the protected list only on a *second* reference, and a cache hit is that
> second reference: a cache with a high hit rate defends itself, one with a low hit rate is correctly
> read as cold and reclaimed. Measured on gpt-oss-120b, where 3000 MiB covers 5.2% of the expert bank
> and returns a 13% hit, the cache is taken back *while decoding* and the fight costs far more than
> the hits are worth. `MemAvailable` also over-states the headroom here, since it counts the page
> cache holding this model's own dense weights as free. Before trusting `auto` on a model whose
> expert set dwarfs the budget, read [android-memory.md](android-memory.md).

> **`auto` deducts pending anon/pinned dense allocations, then sizes once.** The budget is
> chosen in `ExpertStreamSource::init` after the expert-set size is known and *before*
> `DenseWeights::init` allocates. For `Anonymous` and `Pinned`, the same tensors that
> conversion will actually take (skipping oversized tensors that stay mmap'd, and charging
> row-gathered tables only their window) are subtracted from `pio::mem_available_bytes()`
> first, so the cache is not planned as if that RAM were still free. `mmap` and `warm` do
> not convert, so they are not deducted — those pages remain file-backed. `--cache-ceil-mb`
> remains the extra cap; the Android example still ships a 3000 MiB ceiling by default.
> `auto` still does not read cgroup `memory.max`.

> **`auto` sizes from a signal that lies, so keep it modest.** After the anon/pinned
> deduction above, `auto` still reads host `MemAvailable`, which reports memory the
> device will not actually concede (remaining mmap'd weights still count as free, and
> the figure is not a cgroup limit), so it can over-ask — and an over-ask is not a wasted
> budget but a running fight. The runtime governor that once tried to correct this from
> the other end (`--cache-dynamic`) was retired as a net loss (see [pressure.md](pressure.md));
> `auto` now sizes **once at load** and stays fixed, so bound it with `--cache-ceil-mb` on a
> model whose expert set dwarfs the device, or use cache-off.

> **Linux cgroup limits are not an input to `auto`.** The current Linux implementation reads
> host `/proc/meminfo`, not the process's cgroup `memory.max` or `memory.current`. A process
> capped by systemd or a container can therefore receive a cache budget larger than its actual
> memory allowance. The cgroup limit still applies to its allocations: with swap disabled,
> an anonymous-memory working set larger than that allowance can cause an OOM kill rather than an
> automatic reduction of the expert LRU. Use an explicit budget or ceiling with room for dense
> weights, KV and compute buffers when deliberately streaming under such a cap.
>
> For a hot-state experiment, discard a warm-up generation and reuse the same process while
> clearing only KV. Repeating a request can need no expert reads even when the *complete model*
> exceeds the cgroup limit, because only that request's expert working set became resident.
> That zero is cache replay of retained experts, not evidence of small adjacent-token expert
> changes or of zero I/O for long or new text. CPU contention can stretch timings; it cannot
> erase read events that did happen. Clean pages from the initial GGUF mapping can be reclaimed
> separately. If a cap kills the process before warm-up completes, report that case as OOM, not
> as a zero-I/O hot result.

## Flags

| Flag | Meaning |
|---|---|
| `--cache-mb auto` | size the cache to the device instead of a fixed MiB (mutually exclusive with a numeric `--cache-mb`). **The CLI's default whenever `--moe-stream` is on**: pass a number, or `0` for no cache, to override it. The library's own default is still no cache, so an embedder passing 0 keeps meaning it |
| `--cache-floor-mb N` | RAM to leave free for the rest of the system when auto-sizing (default 1536) |
| `--cache-ceil-mb N` | upper bound on the auto-sized budget (0 = no cap). Use it — uncapped `auto` over-asks |
| `--dense-weights mmap\|warm\|anon\|ahwb` | the dense (non-expert) weight policy. `warm` is the load-time page-cache sweep; `mmap` leaves file-backed weights alone; `anon` (default) reads them into anonymous buffers; `ahwb` uses Android-only reclaim-exempt memory. See [benchmarks-gpt-oss.md](benchmarks-gpt-oss.md). `--no-warm-dense` and `--dense-odirect` are legacy aliases for `mmap` and `anon` |

`auto` is a real LRU cache, so it satisfies the cache requirement of `--prefetch`.

## Explicit control

Embedders that link the engine can also resize the cache directly with
`Session::set_cache_budget_mb(int)` — for an app's own memory-pressure callback. It must be called
between generations (never during a decode); it evicts to the new budget immediately. The Android
example does not use it: it runs `bmoe-cli` as a subprocess, so its "Auto" cache setting simply
passes `--cache-mb auto` and the load-time sizing above applies.

## Gate

**S3** proves a runtime resize is byte-safe: it opens a session with a warm cache, drops the budget
to force a full eviction, and asserts the next generation still matches the resident reference —
only residency changes, never the produced bytes.
