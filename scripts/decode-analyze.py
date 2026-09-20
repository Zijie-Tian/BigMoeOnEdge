#!/usr/bin/env python3
"""Read a --compute-trace / --io-trace pair without a spreadsheet.

The per-token CSV says how long a token took and calls the leftover "compute". These traces say
what that leftover is actually made of, and what the flash floor under it really is. Stdlib only.

  decode-analyze.py compute ct.csv          # where compute goes, by op / by layer / faults
  decode-analyze.py io io.csv               # the flash floor: latency, size, waste, lanes
  decode-analyze.py io io.csv --adjacent    # how much of a token's reads could coalesce
  decode-analyze.py timeline ct.csv --io io.csv --svg out.svg --summary out.json

A traced run is not a benchmark run: isolating every node forbids ggml the operator coalescing it
would normally do, and the I/O rows take a lock per read. Read the proportions, not the absolutes.

timeline consumes v2 traces (steady_clock start_ns/end_ns, shared trace_id). It will not invent a
Gantt from v1 latency/order. Parallel I/O and waits are unioned for wall metrics.
"""
import argparse
import csv
import html
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_io import kv_tokens, read_preamble_csv as read_trace

# Frame/callback rows are instrumentation, not ggml ops. Ordinary compute totals must not
# treat a BMOE_DECODE envelope as another kernel, or they double-count the token.
SYNTHETIC_OPS = frozenset(("BMOE_DECODE", "BMOE_CALLBACK"))
EXPERT_MATMUL_OP = "MUL_MAT_ID"
LAYER_OP = "LAYER"

COMPUTE_V2_COLS = (
    "turn", "phase", "step", "seq", "layer", "op", "name", "wall_ns", "majflt",
    "start_ns", "end_ns",
)
IO_V2_COLS = (
    "turn", "phase", "step", "layer", "expert", "proj", "lane", "spec",
    "offset", "req_bytes", "read_bytes", "latency_ns",
    "start_ns", "end_ns", "kind", "thread_id",
)

CAVEATS = [
    "Timestamps are std::chrono::steady_clock nanoseconds from the same process (time_since_epoch), not wall clock.",
    "Graph-node elapsed includes backend scheduling/synchronization and may contain expert-read waits; it is not pure thread CPU time.",
    "Wait union is any-thread unmet-ready (including spin), not all-threads idle.",
    "Read intervals include FileReader copy and alignment; they are not device-only service time.",
    "Cache hits produce no read/wait rows. Parallel I/O/waits are unioned for *union_ms; read_busy_sum_ms sums per-read durations.",
    "MUL_MAT_ID spans are whole graph nodes, not per-expert compute. LAYER rows are coarse segments, not isolated kernels.",
]


def fmt_ms(ns):
    return f"{ns / 1e6:9.2f}"


def bar(frac, width=28):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def decode_only(rows):
    """Steady-state decode: phase 1. Prefill is one batch of many tokens, a different regime."""
    d = [r for r in rows if r["phase"] == "1"]
    return d if d else rows


def io_is_read(row):
    """v2 wait rows must not enter the flash-floor summary; v1 has no kind and is all reads."""
    kind = row.get("kind")
    if kind is None or kind == "":
        return True
    return kind == "read"


def cmd_compute_layers(meta, rows):
    """Layer-granularity trace (--compute-trace-layers): one row per layer segment.

    Fewer barriers than per-node tracing, but still a diagnostic: compare an untraced control.
    "pre" is the embedding lookup, "post" the last layer's tail plus the final norm and LM head.
    """
    steps = sorted({int(r["step"]) for r in rows})
    n_steps = len(steps)
    total = sum(int(r["wall_ns"]) for r in rows)
    faults = sum(int(r["majflt"]) for r in rows)

    print(f"model={meta.get('model','?')} arch={meta.get('arch','?')} "
          f"n_layer={meta.get('n_layer','?')} threads={meta.get('n_threads','?')}")
    print(f"decode steps={n_steps}  layer granularity  "
          f"layer elapsed={fmt_ms(total/max(1,n_steps))} ms/frame  majflt={faults/max(1,n_steps):.0f}/frame")
    print("\nNOTE: segments include graph execution, synchronization and callback/load work.\n"
          "      They are not pure CPU or per-kernel timings; compare an untraced control.\n")

    by_seg = defaultdict(lambda: [0, 0])  # ns, majflt
    for r in rows:
        e = by_seg[r["name"]]
        e[0] += int(r["wall_ns"])
        e[1] += int(r["majflt"])

    def seg_key(name):
        if name == "pre":
            return (-1,)
        if name == "post":
            return (1 << 30,)
        return (int(name.split(".")[1]),) if name.startswith("blk.") else (1 << 29,)

    mx = max(v[0] for v in by_seg.values()) or 1
    print(f"{'segment':<10}{'ms/token':>10}{'share':>8}  {'majflt/tok':>10}")
    for name in sorted(by_seg, key=seg_key):
        ns, mf = by_seg[name]
        print(f"{name:<10}{fmt_ms(ns/max(1,n_steps))}{ns/total*100 if total else 0:7.1f}%  "
              f"{mf/max(1,n_steps):10.1f}  {bar(ns/mx)}")


def cmd_compute(args):
    meta, rows = read_trace(args.path)
    rows = decode_only(rows)
    if not rows:
        sys.exit("no rows")
    rows = [r for r in rows if r.get("op") not in SYNTHETIC_OPS]
    if not rows:
        sys.exit("no rows")
    if all(r["op"] == LAYER_OP for r in rows):
        cmd_compute_layers(meta, rows)
        return
    steps = sorted({int(r["step"]) for r in rows})
    n_steps = len(steps)
    total = sum(int(r["wall_ns"]) for r in rows)
    faults = sum(int(r["majflt"]) for r in rows)

    print(f"model={meta.get('model','?')} arch={meta.get('arch','?')} "
          f"n_layer={meta.get('n_layer','?')} threads={meta.get('n_threads','?')}")
    print(f"decode steps={n_steps}  nodes/step={len(rows)//max(1,n_steps)}  "
          f"measured compute={fmt_ms(total/max(1,n_steps))} ms/token  majflt={faults/max(1,n_steps):.0f}/token")
    print("\nNOTE: measured, not a residual - each node was isolated and synchronized. The barrier\n"
          "      that makes it measurable also inflates it; compare shares, not absolutes.\n")

    # ── by op ──
    by_op = defaultdict(lambda: [0, 0, 0])  # ns, count, majflt
    for r in rows:
        e = by_op[r["op"]]
        e[0] += int(r["wall_ns"])
        e[1] += 1
        e[2] += int(r["majflt"])
    print(f"{'op':<18}{'ms/token':>10}{'share':>8}  {'majflt/tok':>10}  {'':<28}")
    for op, (ns, cnt, mf) in sorted(by_op.items(), key=lambda kv: -kv[1][0])[: args.top]:
        share = ns / total if total else 0
        print(f"{op:<18}{fmt_ms(ns/max(1,n_steps))}{share*100:7.1f}%  {mf/max(1,n_steps):10.1f}  {bar(share)}")

    # ── faults: the question the residual could never answer ──
    if faults:
        print("\nfault attribution - where the >RAM stall is billed as compute")
        by_fault = defaultdict(lambda: [0, 0])
        for r in rows:
            mf = int(r["majflt"])
            if not mf:
                continue
            e = by_fault[r["op"]]
            e[0] += mf
            e[1] += int(r["wall_ns"])
        for op, (mf, ns) in sorted(by_fault.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"  {op:<16}{mf/max(1,n_steps):9.1f} majflt/tok  in {fmt_ms(ns/max(1,n_steps))} ms/tok "
                  f"({ns/total*100:4.1f}% of compute)")
        faulting = sum(v[1] for v in by_fault.values())
        print(f"  {'TOTAL':<16}{faults/max(1,n_steps):9.1f} majflt/tok  in {fmt_ms(faulting/max(1,n_steps))} ms/tok "
              f"({faulting/total*100:4.1f}% of compute)")
        print("  ^ nodes that faulted. Their time is flash wait, not arithmetic - subtract it before\n"
              "    calling this model compute-bound.")

    # ── by layer ──
    if args.layers:
        by_layer = defaultdict(int)
        for r in rows:
            by_layer[int(r["layer"])] += int(r["wall_ns"])
        print("\nby layer (-1 = no layer: embeddings, output head, masks)")
        mx = max(by_layer.values()) or 1
        for il in sorted(by_layer):
            ns = by_layer[il]
            print(f"  layer {il:>3} {fmt_ms(ns/max(1,n_steps))} ms/tok  {bar(ns/mx)}")


def cmd_io(args):
    meta, rows = read_trace(args.path)
    rows = decode_only(rows)
    rows = [r for r in rows if io_is_read(r)]
    if not rows:
        sys.exit("no rows")
    steps = sorted({(int(r.get("turn", 0)), int(r["phase"]), int(r["step"])) for r in rows})
    n_steps = len(steps)
    lat = [int(r["latency_ns"]) for r in rows]
    got = sum(int(r["read_bytes"]) for r in rows)
    want = sum(int(r["req_bytes"]) for r in rows)
    busy = sum(lat)

    print(f"model={meta.get('model','?')} io_threads={meta.get('io_threads','?')} "
          f"o_direct={meta.get('o_direct','?')} overlap={meta.get('overlap','?')}")
    print(f"read-active frames={n_steps}  reads={len(rows)} ({len(rows)/max(1,n_steps):.0f}/active frame)  "
          f"read={got/2**20/max(1,n_steps):.1f} MiB/active frame")
    print("Cache-hit-only frames emit no reads; use timeline with compute frames for full-run per-frame averages.")
    # Reader-service durations include copy/scheduling, not just storage-device work.
    print(f"reader-service bandwidth={got/2**20/(busy/1e9):.0f} MiB/s over summed read durations "
          f"({busy/1e9/max(1,n_steps)*1000:.0f} ms/active frame)")
    waste = (got - want) / got * 100 if got else 0
    print(f"alignment waste={waste:.1f}% ({(got-want)/2**20/max(1,n_steps):.2f} MiB/active frame)")

    # ── latency distribution: is the floor seek-bound or size-bound? ──
    lat.sort()
    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p))] / 1e3
    print("\nper-read latency (us)")
    for p in (0.5, 0.9, 0.99):
        print(f"  p{int(p*100):<3} {pct(p):9.1f}")
    print(f"  max  {lat[-1]/1e3:9.1f}")

    # ── size vs bandwidth: the coalescing case, in one table ──
    print("\nby request size - the per-read cost of scattering")
    buckets = defaultdict(lambda: [0, 0, 0])  # count, bytes, ns
    for r in rows:
        kb = int(r["read_bytes"]) // 1024
        b = 1 << (kb.bit_length() - 1) if kb else 0
        e = buckets[b]
        e[0] += 1
        e[1] += int(r["read_bytes"])
        e[2] += int(r["latency_ns"])
    print(f"  {'size':>8}{'reads':>9}{'MiB':>9}{'MiB/s':>9}{'us/read':>9}")
    for b in sorted(buckets):
        cnt, by, ns = buckets[b]
        print(f"  {b:>6} K{cnt:>9}{by/2**20:9.1f}{by/2**20/(ns/1e9):9.0f}{ns/cnt/1e3:9.1f}")

    # ── lanes ──
    by_lane = defaultdict(lambda: [0, 0])
    for r in rows:
        e = by_lane[int(r["lane"])]
        e[0] += int(r["read_bytes"])
        e[1] += int(r["latency_ns"])
    print("\nby lane - reader-service time includes scheduling and copies")
    for ln in sorted(by_lane):
        by, ns = by_lane[ln]
        print(f"  lane {ln}: {by/2**20/max(1,n_steps):7.1f} MiB/active frame  "
              f"busy {ns/1e9/max(1,n_steps)*1000:7.1f} ms/active frame  {by/2**20/(ns/1e9):6.0f} MiB/s")

    spec = [r for r in rows if r["spec"] == "1"]
    if spec:
        sb = sum(int(r["read_bytes"]) for r in spec)
        print(f"\nspeculative: {len(spec)} reads, {sb/2**20/max(1,n_steps):.1f} MiB/active frame "
              f"({sb/got*100:.0f}% of bytes read)")

    if args.adjacent:
        # How much of a step's reads are back-to-back in the file? That is the ceiling on what an
        # expert-contiguous layout / runtime coalescing could merge — the roadmap's read-bandwidth item.
        print("\nadjacency - the coalescing ceiling")
        merged_tot = runs_tot = 0
        for st in steps:
            ext = sorted((int(r["offset"]), int(r["offset"]) + int(r["read_bytes"]))
                         for r in rows if (int(r.get("turn", 0)), int(r["phase"]), int(r["step"])) == st)
            if not ext:
                continue
            runs, cur_end = 1, ext[0][1]
            for a, b in ext[1:]:
                if a > cur_end:
                    runs += 1
                cur_end = max(cur_end, b)
            merged_tot += len(ext)
            runs_tot += runs
        if merged_tot:
            print(f"  {merged_tot/max(1,n_steps):.0f} reads/active frame span {runs_tot/max(1,n_steps):.0f} "
                  f"contiguous runs/active frame")
            print(f"  perfectly coalesced, that is {merged_tot/max(1,runs_tot):.1f}x fewer requests "
                  f"for the same bytes")


# ── timeline (v2 measured Gantt) ─────────────────────────────────────────────


def ms_json(ns):
    if ns <= 0:
        return 0.0
    return round(ns / 1e6, 6)


def merge_union(spans):
    """Merge (start, end) into disjoint ordered intervals. Touching intervals coalesce."""
    xs = [(s, e) for s, e in spans if e > s]
    if not xs:
        return []
    xs.sort()
    out = [xs[0]]
    for s, e in xs[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def union_len(spans):
    return sum(e - s for s, e in merge_union(spans))


def clip_span(s, e, lo, hi):
    a, b = max(s, lo), min(e, hi)
    return (a, b) if b > a else None


def clip_spans(spans, lo, hi):
    out = []
    for s, e in spans:
        c = clip_span(s, e, lo, hi)
        if c:
            out.append(c)
    return out


def clip_to_windows(spans, windows):
    merged = merge_union(windows)
    out = []
    for s, e in spans:
        for lo, hi in merged:
            c = clip_span(s, e, lo, hi)
            if c:
                out.append(c)
    return out


def intersect_merged(a, b):
    """Intersection of two already-merged unions."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract_merged(a, b):
    """a minus b; both must already be merged unions."""
    if not a:
        return []
    if not b:
        return list(a)
    out = []
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            bs, be = b[k]
            if bs > cur:
                out.append((cur, min(bs, e)))
            if be > cur:
                cur = be
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return [(s, e) for s, e in out if e > s]


def clipped_duration_sum(spans, windows):
    """Sum (not union) of span durations clipped to the union of windows."""
    tot = 0
    for s, e in clip_to_windows(spans, windows):
        tot += e - s
    return tot


def bytes_overlapping(items, windows):
    """Sum nbytes for each (start, end, nbytes) that overlaps any window. Each item once."""
    merged = merge_union(windows)
    tot = 0
    for s, e, n in items:
        hit = False
        for lo, hi in merged:
            if e > s:
                if max(s, lo) < min(e, hi):
                    hit = True
                    break
            elif lo <= s < hi:
                hit = True
                break
        if hit:
            tot += n
    return tot


def load_preamble_csv(path):
    meta, kind_line, body = {}, None, []
    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#"):
                stripped = ln.lstrip("#").strip()
                if stripped.startswith("compute_trace") or stripped.startswith("io_trace"):
                    kind_line = stripped
                meta.update(kv_tokens(ln.rstrip()))
            else:
                body.append(ln)
    reader = csv.DictReader(body)
    fields = list(reader.fieldnames or [])
    return meta, list(reader), fields, kind_line


def _kind_parts(kind_line):
    if not kind_line:
        return None, None
    parts = kind_line.split()
    name = parts[0] if parts else None
    ver = None
    for p in parts[1:]:
        if p.startswith("v") and p[1:].isdigit():
            ver = p
            break
    return name, ver


def parse_u64(row, key, where):
    if key not in row or row[key] is None or row[key] == "":
        sys.exit(f"timeline: {where}: missing {key} (v2 timestamps required; "
                 f"refusing to reconstruct chronology from latency/order)")
    try:
        v = int(row[key])
    except ValueError:
        sys.exit(f"timeline: {where}: {key}={row[key]!r} is not an integer")
    if v < 0:
        sys.exit(f"timeline: {where}: {key} is negative")
    return v


def row_span(row, dur_key, where):
    start = parse_u64(row, "start_ns", where)
    end = parse_u64(row, "end_ns", where)
    if start > end:
        sys.exit(f"timeline: {where}: start_ns ({start}) > end_ns ({end})")
    dur = end - start
    stated = parse_u64(row, dur_key, where)
    if stated != dur:
        sys.exit(f"timeline: {where}: {dur_key}={stated} != end_ns-start_ns={dur}")
    return start, end


def parse_trace_id(meta, path):
    tid = meta.get("trace_id")
    if tid is None or tid == "":
        sys.exit(f"timeline: {path}: missing trace_id; refusing to pair files by name or mtime")
    try:
        n = int(tid)
    except ValueError:
        return tid
    if n == 0:
        sys.exit(f"timeline: {path}: trace_id=0 is not a session id")
    return str(n)


def require_v2(path, kind_line, meta, fields, expect_kind, required_cols):
    name, ver = _kind_parts(kind_line)
    if name and name != expect_kind:
        sys.exit(f"timeline: {path}: preamble {kind_line!r} is not a {expect_kind} file")
    if ver and ver != "v2":
        sys.exit(f"timeline: {path} is {kind_line!r}, not v2; "
                 f"refusing to reconstruct v1 chronology from latency/order")
    if meta.get("clock") != "steady_ns":
        sys.exit(f"timeline: {path}: clock={meta.get('clock')!r} (need clock=steady_ns); "
                 f"refusing to mix epochs")
    parse_trace_id(meta, path)
    missing = [c for c in required_cols if c not in fields]
    if missing:
        sys.exit(f"timeline: {path}: missing v2 columns {missing}; "
                 f"refusing to reconstruct chronology from latency/order")
    if ver is None and "start_ns" not in fields:
        sys.exit(f"timeline: {path}: not a v2 trace (no version line, no start_ns)")


def where_compute(path, r, i):
    return (f"{path} row {i} turn={r.get('turn')} phase={r.get('phase')} "
            f"step={r.get('step')} seq={r.get('seq')} op={r.get('op')}")


def where_io(path, r, i):
    return (f"{path} row {i} turn={r.get('turn')} phase={r.get('phase')} "
            f"step={r.get('step')} kind={r.get('kind')} lane={r.get('lane')}")


def parse_compute_v2(path):
    meta, rows, fields, kind_line = load_preamble_csv(path)
    require_v2(path, kind_line, meta, fields, "compute_trace", COMPUTE_V2_COLS)
    events = []
    frames = []
    seen_frame = {}
    for i, r in enumerate(rows):
        loc = where_compute(path, r, i)
        start, end = row_span(r, "wall_ns", loc)
        parse_u64(r, "majflt", loc)
        op = r.get("op") or ""
        ev = {
            "turn": int(r["turn"]),
            "phase": int(r["phase"]),
            "step": int(r["step"]),
            "seq": int(r["seq"]),
            "layer": int(r["layer"]),
            "op": op,
            "name": r.get("name") or "",
            "start": start,
            "end": end,
            "wall_ns": end - start,
            "majflt": int(r["majflt"]),
        }
        if op == "BMOE_DECODE":
            key = (ev["turn"], ev["phase"], ev["step"])
            if key in seen_frame:
                sys.exit(f"timeline: {loc}: duplicate BMOE_DECODE for turn/phase/step {key}")
            seen_frame[key] = True
            frames.append(ev)
        else:
            events.append(ev)
    frames.sort(key=lambda e: (e["turn"], e["phase"], e["step"], e["start"]))
    if not frames:
        sys.exit(f"timeline: {path}: no BMOE_DECODE frame rows; "
                 f"cannot select a measured decode frame "
                 f"(refusing to reconstruct chronology from node order)")
    return meta, events, frames


def parse_io_v2(path):
    meta, rows, fields, kind_line = load_preamble_csv(path)
    require_v2(path, kind_line, meta, fields, "io_trace", IO_V2_COLS)
    reads, waits = [], []
    for i, r in enumerate(rows):
        loc = where_io(path, r, i)
        start, end = row_span(r, "latency_ns", loc)
        kind = r.get("kind") or ""
        if kind not in ("read", "wait"):
            sys.exit(f"timeline: {loc}: kind must be 'read' or 'wait', not {kind!r}")
        parse_u64(r, "thread_id", loc)
        ev = {
            "turn": int(r["turn"]),
            "phase": int(r["phase"]),
            "step": int(r["step"]),
            "layer": int(r["layer"]),
            "expert": int(r["expert"]),
            "proj": int(r["proj"]),
            "lane": int(r["lane"]),
            "spec": int(r["spec"]) if r.get("spec") not in (None, "") else 0,
            "offset": parse_u64(r, "offset", loc),
            "req_bytes": parse_u64(r, "req_bytes", loc),
            "read_bytes": parse_u64(r, "read_bytes", loc),
            "start": start,
            "end": end,
            "thread_id": int(r["thread_id"]),
            "kind": kind,
        }
        if kind == "read":
            reads.append(ev)
        else:
            waits.append(ev)
    return meta, reads, waits


def select_frames(frames, turn, phase, step):
    phase_used = 1 if phase is None else phase
    cands = frames
    if turn is not None:
        cands = [f for f in cands if f["turn"] == turn]
    cands = [f for f in cands if f["phase"] == phase_used]
    if step is not None:
        cands = [f for f in cands if f["step"] == step]
    if not cands:
        bits = [f"phase={phase_used}"]
        if turn is not None:
            bits.append(f"turn={turn}")
        if step is not None:
            bits.append(f"step={step}")
        hint = ""
        if phase is None and phase_used == 1 and any(f["phase"] == 0 for f in frames):
            hint = " (prefill frames exist; pass --phase 0)"
        sys.exit("timeline: no BMOE_DECODE frame for " + ", ".join(bits) + hint)
    return phase_used, cands


def empty_metrics(frames_n):
    return {
        "decode_wall_ms": 0.0,
        "graph_wall_ms": 0.0,
        "expert_matmul_wall_ms": 0.0,
        "expert_wait_union_ms": 0.0,
        "graph_nonwait_ms": 0.0,
        "callback_wall_ms": 0.0,
        "read_busy_union_ms": 0.0,
        "read_busy_sum_ms": 0.0,
        "read_graph_nonwait_overlap_ms": 0.0,
        "read_bytes": 0,
        "other_wall_ms": 0.0,
        "frames": frames_n,
    }


def compute_metrics(windows, graph, expert, callback, waits, reads):
    """Interval-union metrics over decode windows. Spans are (start, end) ns; reads also carry bytes."""
    if not windows:
        return empty_metrics(0)
    win = merge_union((w["start"], w["end"]) if isinstance(w, dict) else w for w in windows)
    g = merge_union(clip_to_windows(graph, win))
    e = merge_union(clip_to_windows(expert, win))
    c = merge_union(clip_to_windows(callback, win))
    w = merge_union(clip_to_windows(waits, win))
    r_spans = [(rd["start"], rd["end"]) if isinstance(rd, dict) else rd[:2] for rd in reads]
    r = merge_union(clip_to_windows(r_spans, win))
    gnw = subtract_merged(g, w)
    gc = merge_union(g + c)
    other = subtract_merged(win, gc)
    byte_items = []
    for rd in reads:
        if isinstance(rd, dict):
            byte_items.append((rd["start"], rd["end"], rd["read_bytes"]))
        else:
            byte_items.append((rd[0], rd[1], rd[2]))
    return {
        "decode_wall_ms": ms_json(union_len(win)),
        "graph_wall_ms": ms_json(union_len(g)),
        "expert_matmul_wall_ms": ms_json(union_len(e)),
        "expert_wait_union_ms": ms_json(union_len(w)),
        "graph_nonwait_ms": ms_json(union_len(gnw)),
        "callback_wall_ms": ms_json(union_len(c)),
        "read_busy_union_ms": ms_json(union_len(r)),
        "read_busy_sum_ms": ms_json(clipped_duration_sum(r_spans, win)),
        "read_graph_nonwait_overlap_ms": ms_json(union_len(intersect_merged(r, gnw))),
        "read_bytes": bytes_overlapping(byte_items, win),
        "other_wall_ms": ms_json(union_len(other)),
        "frames": len(windows),
    }


def pack_lanes(items):
    """Greedy interval packing. items have start/end. Returns n_lanes and items with lane_i."""
    ordered = sorted(items, key=lambda x: (x["start"], x["end"]))
    ends = []
    for it in ordered:
        placed = False
        for i, e in enumerate(ends):
            if it["start"] >= e:
                ends[i] = it["end"]
                it["lane_i"] = i
                placed = True
                break
        if not placed:
            it["lane_i"] = len(ends)
            ends.append(it["end"])
    return max(1, len(ends)), ordered


def xml_esc(s):
    return html.escape(str(s), quote=True)


def fmt_axis_ms(ns_from_origin):
    ms = ns_from_origin / 1e6
    if abs(ms) < 0.001 and ns_from_origin != 0:
        return f"{ms:.6f}"
    if abs(ms) < 1:
        return f"{ms:.3f}"
    if abs(ms) < 100:
        return f"{ms:.2f}"
    return f"{ms:.1f}"


def layer_zoom_window(frame, layer, graph_ev, cb_ev, reads, waits):
    lo, hi = frame["start"], frame["end"]
    hits = []
    for ev in graph_ev + cb_ev:
        if ev["layer"] == layer:
            c = clip_span(ev["start"], ev["end"], lo, hi)
            if c:
                hits.append(c)
    for ev in reads + waits:
        if ev["layer"] == layer:
            c = clip_span(ev["start"], ev["end"], lo, hi)
            if c:
                hits.append(c)
    if not hits:
        sys.exit(f"timeline: no events for --layer {layer} inside the selected frame "
                 f"(turn={frame['turn']} phase={frame['phase']} step={frame['step']})")
    return min(s for s, _ in hits), max(e for _, e in hits)


def wrap_text(text, max_chars):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def render_svg(path, *, meta, plot, metrics, graph_ev, cb_ev, reads, waits,
               layer_mode, zoom_layer, view_t0, view_t1, caveats):
    frame_t0, frame_t1 = plot["start"], plot["end"]
    gutter = 308
    right = 16
    width = 1400
    plot_w = width - gutter - right
    lane_h = 38
    lane_gap = 4
    metrics_w = 268

    COL_EXPERT = "#E69F00"
    COL_GRAPH = "#0072B2"
    COL_LAYER = "#56B4E9"
    COL_CB = "#009E73"
    COL_READ = "#CC79A7"
    COL_WAIT = "#666666"
    COL_SPEC = "#882255"

    def x_of(t):
        span = view_t1 - view_t0
        if span <= 0:
            return gutter
        return gutter + (t - view_t0) * plot_w / span

    def clip_bar(s, e):
        return clip_span(s, e, view_t0, view_t1)

    lanes = []

    graph_items = []
    for ev in graph_ev:
        c = clip_bar(ev["start"], ev["end"])
        if not c:
            continue
        expert = ev["op"] == EXPERT_MATMUL_OP
        if layer_mode:
            fill = COL_LAYER
            short = ev["name"] or LAYER_OP
        elif expert:
            fill = COL_EXPERT
            short = "MUL_MAT_ID"
        else:
            fill = COL_GRAPH
            short = ev["op"] or ev["name"]
        title = (f"op={ev['op']} name={ev['name']} layer={ev['layer']} seq={ev['seq']} "
                 f"wall_ms={(ev['end']-ev['start'])/1e6:.6f} majflt={ev['majflt']} "
                 f"start_ns={ev['start']} end_ns={ev['end']}")
        graph_items.append({
            "start": c[0], "end": c[1], "fill": fill, "text": short, "title": title,
            "dash": False,
        })
    n_gl, graph_items = pack_lanes(graph_items) if graph_items else (1, [])
    if layer_mode:
        g0 = ["Layer segments", "coarse; not isolated kernels", "includes sync/sched"]
    else:
        g0 = ["Graph nodes", "elapsed includes sync/sched", "may contain waits"]
    for i in range(n_gl):
        bars = [b for b in graph_items if b.get("lane_i") == i]
        lanes.append({"lines": g0 if i == 0 else ["Graph nodes (overlap)"], "bars": bars})

    cb_items = []
    for ev in cb_ev:
        c = clip_bar(ev["start"], ev["end"])
        if not c:
            continue
        title = (f"op=BMOE_CALLBACK name={ev['name']} layer={ev['layer']} "
                 f"wall_ms={(ev['end']-ev['start'])/1e6:.6f} start_ns={ev['start']} end_ns={ev['end']}")
        cb_items.append({
            "start": c[0], "end": c[1], "fill": COL_CB,
            "text": ev["name"] or "callback", "title": title, "dash": False,
        })
    n_cl, cb_items = pack_lanes(cb_items) if cb_items else (1, [])
    for i in range(n_cl):
        bars = [b for b in cb_items if b.get("lane_i", 0) == i]
        lines = ["Callback / cache-load", "ask/observe; not kernel"] if i == 0 else ["Callback (overlap)"]
        lanes.append({"lines": lines, "bars": bars})

    by_lane = defaultdict(list)
    for ev in reads:
        c = clip_bar(ev["start"], ev["end"])
        if not c:
            continue
        spec = ev["spec"] == 1
        label = f"L{ev['layer']} E{ev['expert']} p{ev['proj']}"
        title = (f"kind=read lane={ev['lane']} layer={ev['layer']} expert={ev['expert']} "
                 f"proj={ev['proj']} spec={ev['spec']} req_bytes={ev['req_bytes']} "
                 f"read_bytes={ev['read_bytes']} thread_id={ev['thread_id']} "
                 f"latency_ms={(ev['end']-ev['start'])/1e6:.6f} "
                 f"start_ns={ev['start']} end_ns={ev['end']} offset={ev['offset']}")
        by_lane[ev["lane"]].append({
            "start": c[0], "end": c[1], "fill": COL_SPEC if spec else COL_READ,
            "text": label, "title": title, "dash": spec,
        })
    if by_lane:
        for ln in sorted(by_lane):
            items = by_lane[ln]
            n_il, items = pack_lanes(items)
            for i in range(n_il):
                bars = [b for b in items if b["lane_i"] == i]
                head = f"I/O lane {ln}"
                sub = "FileReader copy+align; not device-only"
                lines = [head, sub] if i == 0 else [f"I/O lane {ln} (overlap)"]
                lanes.append({"lines": lines, "bars": bars})
    else:
        lanes.append({
            "lines": ["I/O (no kind=read in window)", "cache hits emit no read rows"],
            "bars": [],
        })

    wait_merged = merge_union(
        c for ev in waits if (c := clip_bar(ev["start"], ev["end"]))
    )
    wait_bars = []
    for s, e in wait_merged:
        wait_bars.append({
            "start": s, "end": e, "fill": COL_WAIT,
            "text": "wait",
            "title": (f"kind=wait union (any-thread unmet-ready, including spin; "
                      f"not all-threads idle) wall_ms={(e-s)/1e6:.6f} "
                      f"start_ns={s} end_ns={e}"),
            "dash": False,
        })
    lanes.append({
        "lines": ["Any-worker wait union", "any thread unmet-ready", "not all-threads idle"],
        "bars": wait_bars,
    })

    metric_lines = [
        ("decode_wall_ms", metrics["decode_wall_ms"]),
        ("graph_wall_ms", metrics["graph_wall_ms"]),
        ("expert_matmul_wall_ms", metrics["expert_matmul_wall_ms"]),
        ("expert_wait_union_ms", metrics["expert_wait_union_ms"]),
        ("graph_nonwait_ms", metrics["graph_nonwait_ms"]),
        ("callback_wall_ms", metrics["callback_wall_ms"]),
        ("read_busy_union_ms", metrics["read_busy_union_ms"]),
        ("read_busy_sum_ms", metrics["read_busy_sum_ms"]),
        ("read_graph_nonwait_overlap_ms", metrics["read_graph_nonwait_overlap_ms"]),
        ("other_wall_ms", metrics["other_wall_ms"]),
        ("read_bytes", metrics["read_bytes"]),
    ]
    header_h = 22 + 13 * (1 + len(metric_lines)) + 16
    top = max(108, header_h)
    plot_h = len(lanes) * (lane_h + lane_gap) - lane_gap
    span = view_t1 - view_t0

    ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = view_t0 + int(round(span * frac)) if span > 0 else view_t0
        ticks.append((t, fmt_axis_ms(t - frame_t0)))

    title = (f"Measured decode timeline (not schematic)  "
             f"turn={plot['turn']} phase={plot['phase']} step={plot['step']}  "
             f"frame {fmt_axis_ms(frame_t1 - frame_t0)} ms")
    subtitle = (f"model={meta.get('model', '?')} arch={meta.get('arch', '?')}  "
                f"clock=steady_ns trace_id={meta.get('trace_id', '?')}  "
                f"n_threads={meta.get('n_threads', '?')} io_threads={meta.get('io_threads', '?')} "
                f"overlap={meta.get('overlap', '?')} o_direct={meta.get('o_direct', '?')}")
    zoom_note = ""
    if zoom_layer is not None:
        zoom_note = (f"x-axis zoomed to layer {zoom_layer} "
                     f"({fmt_axis_ms(view_t0 - frame_t0)}-{fmt_axis_ms(view_t1 - frame_t0)} ms "
                     f"from frame start); summary numbers are the whole frame, not this zoom.")
    elif layer_mode:
        zoom_note = "LAYER granularity: bars are coarse layer segments, not per-kernel compute."

    caveat_lines = []
    for cav in caveats:
        caveat_lines.extend(wrap_text(cav, 155))
    axis_y = top + plot_h + 4
    height = axis_y + 36 + 12 * len(caveat_lines) + 16

    out = []
    a = out.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
      f'viewBox="0 0 {width} {height}">')
    a("<desc>Measured load/compute Gantt from v2 compute/io traces. "
      "Bar widths are end_ns-start_ns on a shared steady_clock axis.</desc>")
    a("<defs>")
    a('<pattern id="specHatch" patternUnits="userSpaceOnUse" width="6" height="6">')
    a(f'<rect width="6" height="6" fill="{COL_READ}"/>')
    a('<path d="M0,6 L6,0" stroke="#fff" stroke-width="1"/>')
    a("</pattern>")
    a("</defs>")
    a('<rect width="100%" height="100%" fill="#ffffff"/>')
    a(f'<text x="16" y="22" font-family="DejaVu Sans, Liberation Sans, sans-serif" '
      f'font-size="15" font-weight="700" fill="#111">{xml_esc(title)}</text>')
    a(f'<text x="16" y="40" font-family="DejaVu Sans, Liberation Sans, sans-serif" '
      f'font-size="11" fill="#333">{xml_esc(subtitle)}</text>')
    if zoom_note:
        a(f'<text x="16" y="56" font-family="DejaVu Sans, Liberation Sans, sans-serif" '
          f'font-size="11" fill="#333">{xml_esc(zoom_note)}</text>')

    legend = [
        (COL_EXPERT, "expert matmul (MUL_MAT_ID)"),
        (COL_LAYER if layer_mode else COL_GRAPH,
         "layer segment (coarse)" if layer_mode else "other graph node"),
        (COL_CB, "callback / cache-load"),
        (COL_READ, "I/O read (FileReader)"),
        (COL_WAIT, "any-worker wait union"),
    ]
    lx, ly = 16, 78
    a('<g font-family="DejaVu Sans, Liberation Sans, sans-serif" font-size="10" fill="#222">')
    for col, lab in legend:
        a(f'<rect x="{lx}" y="{ly - 9}" width="12" height="12" fill="{col}"/>')
        a(f'<text x="{lx + 16}" y="{ly}">{xml_esc(lab)}</text>')
        lx += 28 + len(lab) * 6 + 14
        if lx > width - metrics_w - 180:
            lx = 16
            ly += 16
    a("</g>")

    mx = width - metrics_w - 8
    my = 18
    a('<g font-family="DejaVu Sans, Liberation Sans, sans-serif" font-size="10" fill="#111">')
    a(f'<text x="{mx}" y="{my}" font-weight="700">plot-frame metrics (ms; bytes raw)</text>')
    for i, (k, v) in enumerate(metric_lines):
        yy = my + 13 * (i + 1)
        val = f"{v}" if k == "read_bytes" else f"{v:.3f}"
        a(f'<text x="{mx}" y="{yy}">{xml_esc(k)}</text>')
        a(f'<text x="{mx + metrics_w - 8}" y="{yy}" text-anchor="end">{xml_esc(val)}</text>')
    a("</g>")

    font = "DejaVu Sans, Liberation Sans, sans-serif"
    a(f'<g font-family="{font}">')
    y = top
    for li, lane in enumerate(lanes):
        band = "#f4f6f8" if li % 2 == 0 else "#ffffff"
        a(f'<rect x="{gutter}" y="{y}" width="{plot_w}" height="{lane_h}" fill="{band}"/>')
        y += lane_h + lane_gap
    for t, lab in ticks:
        x = x_of(t)
        a(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
          f'stroke="#dde1e6" stroke-width="1"/>')
    y = top
    for lane in lanes:
        lines = lane["lines"]
        a(f'<text x="12" y="{y + 12}" font-size="10" font-weight="700" fill="#111">'
          f'{xml_esc(lines[0])}</text>')
        for j, extra in enumerate(lines[1:]):
            a(f'<text x="12" y="{y + 12 + 11 * (j + 1)}" font-size="9" fill="#555">'
              f'{xml_esc(extra)}</text>')
        for b in lane["bars"]:
            x1 = x_of(b["start"])
            x2 = x_of(b["end"])
            bw = x2 - x1
            fill = "url(#specHatch)" if b.get("dash") else b["fill"]
            dash = ' stroke="#441122" stroke-dasharray="3,2"' if b.get("dash") else ' stroke="none"'
            a(f'<rect x="{x1:.2f}" y="{y + 6}" width="{bw:.2f}" height="{lane_h - 12}" '
              f'fill="{fill}"{dash}>')
            a(f'<title>{xml_esc(b["title"])}</title>')
            a("</rect>")
            if bw >= 44 and b.get("text"):
                a(f'<text x="{x1 + 3:.2f}" y="{y + lane_h / 2 + 4:.1f}" font-size="9" fill="#fff">'
                  f'{xml_esc(b["text"][: max(1, int(bw / 7))])}</text>')
        y += lane_h + lane_gap
    a("</g>")

    a(f'<line x1="{gutter}" y1="{axis_y}" x2="{gutter + plot_w}" y2="{axis_y}" '
      f'stroke="#222" stroke-width="1"/>')
    a(f'<g font-family="{font}" font-size="10" fill="#222">')
    for t, lab in ticks:
        x = x_of(t)
        a(f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y + 5}" stroke="#222"/>')
        a(f'<text x="{x:.2f}" y="{axis_y + 16}" text-anchor="middle">{xml_esc(lab)}</text>')
    axis_label = "time (ms from frame start; steady_clock; bar width = end_ns-start_ns, no padding)"
    a(f'<text x="{gutter + plot_w / 2:.1f}" y="{axis_y + 30}" text-anchor="middle" font-size="10" fill="#444">'
      f'{xml_esc(axis_label)}</text>')
    a("</g>")

    cy = axis_y + 46
    a(f'<g font-family="{font}" font-size="9" fill="#444">')
    for i, line in enumerate(caveat_lines):
        a(f'<text x="16" y="{cy + i * 12}">{xml_esc(line)}</text>')
    a("</g>")
    a("</svg>\n")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))



def write_summary_json(path, obj):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def cmd_timeline(args):
    c_meta, c_events, frames = parse_compute_v2(args.compute)
    i_meta, reads, waits = parse_io_v2(args.io)

    c_tid = parse_trace_id(c_meta, args.compute)
    i_tid = parse_trace_id(i_meta, args.io)
    if c_tid != i_tid:
        sys.exit(f"timeline: trace_id mismatch compute={c_tid} io={i_tid}; "
                 f"refusing to pair different runs")

    phase_used, selected = select_frames(frames, args.turn, args.phase, args.step)
    plot = selected[0]
    summary_frames = [plot] if args.step is not None else selected

    graph_all = [e for e in c_events if e["op"] != "BMOE_CALLBACK"]
    cb_all = [e for e in c_events if e["op"] == "BMOE_CALLBACK"]
    layer_mode = bool(graph_all) and all(e["op"] == LAYER_OP for e in graph_all)

    def spans(evs, pred=None):
        return [(e["start"], e["end"]) for e in evs if pred is None or pred(e)]

    graph_spans = spans(graph_all)
    expert_spans = spans(graph_all, lambda e: e["op"] == EXPERT_MATMUL_OP)
    cb_spans = spans(cb_all)
    wait_spans = spans(waits)

    per_frame = []
    for fr in summary_frames:
        m = compute_metrics(
            [fr], graph_spans, expert_spans, cb_spans, wait_spans, reads,
        )
        per_frame.append({
            "turn": fr["turn"],
            "phase": fr["phase"],
            "step": fr["step"],
            "start_ns": fr["start"],
            "end_ns": fr["end"],
            **m,
        })

    aggregate = compute_metrics(
        summary_frames, graph_spans, expert_spans, cb_spans, wait_spans, reads,
    )
    plot_metrics = compute_metrics(
        [plot], graph_spans, expert_spans, cb_spans, wait_spans, reads,
    )

    view_t0, view_t1 = plot["start"], plot["end"]
    if args.layer is not None:
        view_t0, view_t1 = layer_zoom_window(
            plot, args.layer, graph_all, cb_all, reads, waits,
        )

    units = {
        "time_metrics": "milliseconds (interval union unless the name is read_busy_sum_ms)",
        "read_busy_sum_ms": "milliseconds (sum of per-read durations; parallel lanes exceed wall)",
        "read_bytes": "bytes actually read (aligned window), counted once per overlapping read",
        "timestamps": "nanoseconds, std::chrono::steady_clock::time_since_epoch(), same process",
    }
    selection = {
        "turn": args.turn,
        "phase": phase_used,
        "step": args.step,
        "layer_zoom": args.layer,
        "summary_scope": "selected_frame" if args.step is not None else "phase_aggregate",
        "plot": {
            "turn": plot["turn"],
            "phase": plot["phase"],
            "step": plot["step"],
            "start_ns": plot["start"],
            "end_ns": plot["end"],
            "view_start_ns": view_t0,
            "view_end_ns": view_t1,
        },
    }
    report = {
        "clock": "steady_ns",
        "trace_id": c_tid,
        "model": c_meta.get("model"),
        "arch": c_meta.get("arch"),
        "granularity": "layer" if layer_mode else "node",
        "units": units,
        "selection": selection,
        "caveats": CAVEATS,
        "plot_frame": plot_metrics,
        "aggregate": aggregate,
        "per_frame": per_frame,
    }

    # Clip plotted events to the view window by timestamp (shared epoch), not by CSV step.
    render_svg(
        args.svg,
        meta=c_meta,
        plot=plot,
        metrics=plot_metrics,
        graph_ev=graph_all,
        cb_ev=cb_all,
        reads=reads,
        waits=waits,
        layer_mode=layer_mode,
        zoom_layer=args.layer,
        view_t0=view_t0,
        view_t1=view_t1,
        caveats=CAVEATS,
    )
    write_summary_json(args.summary, report)

    print(f"plotted turn={plot['turn']} phase={plot['phase']} step={plot['step']}  "
          f"decode_wall_ms={plot_metrics['decode_wall_ms']}  "
          f"granularity={report['granularity']}")
    print(f"summary scope={selection['summary_scope']} frames={aggregate['frames']}  "
          f"decode_wall_ms={aggregate['decode_wall_ms']}")
    print(f"wrote {args.svg}")
    print(f"wrote {args.summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="what the compute residual is actually made of")
    c.add_argument("path")
    c.add_argument("--top", type=int, default=12, help="how many ops to list")
    c.add_argument("--layers", action="store_true", help="also break down by layer")
    c.set_defaults(fn=cmd_compute)

    i = sub.add_parser("io", help="the flash floor: latency, size, waste, lanes")
    i.add_argument("path")
    i.add_argument("--adjacent", action="store_true", help="estimate the coalescing ceiling")
    i.set_defaults(fn=cmd_io)

    t = sub.add_parser(
        "timeline",
        help="measured load/compute Gantt from v2 traces (union/intersection, not schematic)",
    )
    t.add_argument("compute", help="v2 --compute-trace CSV")
    t.add_argument("--io", required=True, help="v2 --io-trace CSV (same trace_id)")
    t.add_argument("--svg", required=True, help="measured Gantt output")
    t.add_argument("--summary", required=True, help="JSON interval-union metrics")
    t.add_argument("--step", type=int, help="plot and limit summary to this decode step")
    t.add_argument("--phase", type=int, choices=(0, 1), help="0=prefill 1=decode (default: 1)")
    t.add_argument("--turn", type=int, help="session turn (default: first matching phase)")
    t.add_argument("--layer", type=int,
                   help="plot x-axis zoom to this layer; does not change summary numbers")
    t.set_defaults(fn=cmd_timeline)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
