// Decode traces: what a token's time is actually made of.
//
// The per-token metrics answer *how long*; the route trace answers *what the router asked for*.
// These two answer *where the time went*, and they exist because the headline number they
// decompose is not measured at all: `compute_ms` is a residual (wall − io − mgmt), so every cost
// the engine does not itself clock — page faults, scheduler stalls, the matmuls themselves — is
// silently pooled into it. A residual cannot tell you which of those it is.
//
//   Compute trace — the eval callback, asked to isolate nodes, yields real per-node wall time:
//   ggml computes exactly up to an isolated node, synchronizes, then calls back. v2 timestamps
//   (start_ns/end_ns) are std::chrono::steady_clock::time_since_epoch nanoseconds from this
//   process. In per-node mode a node's span begins AFTER ask-side callback work and ends BEFORE
//   observe-side callback work, so serial load/cache work is a BMOE_CALLBACK row, not the next
//   kernel. Node wall includes backend scheduling, synchronization, and expert-read waits; it is
//   not pure thread CPU time. BMOE_DECODE is the whole begin/end_compute_batch frame.
//
//   I/O trace — one row per FileReader::read (kind=read: copy/alignment included) and one row per
//   compute-thread unmet-ready interval (kind=wait: spin included). Cache hits produce neither.
//   Wait rows have lane=-1 and zero byte counts. latency_ns = end_ns - start_ns.
//
// Both are diagnostics, not telemetry, and both perturb what they measure — isolating nodes
// forbids ggml the operator coalescing it would otherwise do, and the I/O rows take a lock on the
// read path. A traced run is NOT a benchmark run: read the proportions, not the absolutes.
//
//   Layer granularity — the compute trace can instead isolate only the FIRST node of each layer
//   (RunConfig::compute_trace_layers). ~n_layer barriers per token instead of ~3000 preserves
//   operator coalescing and, crucially, the async expert prefetch: the io lanes keep streaming
//   across a boundary, so the numbers stay close to an untraced run. Rows carry op "LAYER" and
//   aggregate everything since the previous boundary (callback work included — broader than
//   per-node): name "blk.<il>" is layer il's segment, "pre" is the embedding lookup before layer
//   0, and "post" (emitted when the batch closes) is the last layer's tail plus the final norm
//   and LM head — per-op detail inside a segment is what this mode trades away.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bmoe {

// Monotonic nanoseconds of steady_clock::time_since_epoch in this process. Never wall clock.
inline uint64_t decode_trace_now_ns() noexcept {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// One isolated graph node's compute, a LAYER segment, a BMOE_CALLBACK interval, or the
// BMOE_DECODE frame. Emitted only while the compute trace is on.
struct ComputeTraceRow {
    int turn = 0;  // session-mode turn (0 for a one-shot run)
    int phase = 0; // 0 = prefill, 1 = decode
    int step = 0;  // absolute context position of the token being computed
    int seq = 0;   // trace-event order in this decode, including frame/callback rows
    int layer = -1;
    // ggml's op name (ggml_op_name) and the node's own name, or BMOE_DECODE / BMOE_CALLBACK / LAYER.
    // Deliberately raw: which node belongs to attention vs the dense FFN vs the expert matmul is
    // naming policy that varies by architecture, so the engine reports what the graph said and
    // the analysis script classifies.
    std::string op;
    std::string name;
    uint64_t wall_ns = 0;  // end_ns - start_ns (node wall is not pure thread CPU time)
    uint64_t majflt = 0;   // major page faults charged to this interval
    uint64_t start_ns = 0; // decode_trace_now_ns() at interval open
    uint64_t end_ns = 0;   // decode_trace_now_ns() at interval close
};

// One flash read (kind="read") or one compute-thread wait on an unmet ready flag (kind="wait").
// Emitted only while the I/O trace is on.
struct IoTraceRow {
    int turn = 0;
    int phase = 0;
    int step = 0;
    int layer = -1;
    int32_t expert = -1;
    int8_t proj = -1;        // projection slot within the layer (recipe order)
    int8_t lane = -1;        // which read lane served it; -1 for kind=wait
    uint8_t spec = 0;        // 1 if issued speculatively by prefetch
    uint64_t offset = 0;     // absolute file offset requested
    uint64_t req_bytes = 0;  // bytes the caller wanted
    uint64_t read_bytes = 0; // bytes actually read (aligned window; ≥ req_bytes with O_DIRECT)
    uint64_t latency_ns = 0; // end_ns - start_ns
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    const char * kind = "read"; // "read" or "wait"; points at a string literal
    uint64_t thread_id = 0;     // stable in-process numeric id (hash of std::thread::id)
};

// Run-level facts, emitted once before any row.
struct DecodeTraceStatic {
    std::string model;
    std::string arch;
    int n_layer = 0;
    int n_threads = 0;
    int io_threads = 0;
    bool o_direct = false;
    bool overlap = false;
    uint64_t trace_id = 0; // shared by both sinks of one Session::open; nonzero
};

// Optional sinks. The engine calls on_static once at open, then on_rows once per decode with that
// decode's rows. Rows are buffered in RAM while the graph runs — the callback and the read path
// must not do I/O — and drained after llama_decode returns.
class IComputeTraceSink {
public:
    virtual ~IComputeTraceSink() = default;
    virtual void on_static(const DecodeTraceStatic &) = 0;
    virtual void on_rows(const ComputeTraceRow * rows, size_t n) = 0;
};

class IIoTraceSink {
public:
    virtual ~IIoTraceSink() = default;
    virtual void on_static(const DecodeTraceStatic &) = 0;
    virtual void on_rows(const IoTraceRow * rows, size_t n) = 0;
};

// Sinks writing `path` as long-format CSV: a `#` preamble carrying the static block, then one row
// per node / per read. Return nullptr if the file cannot be opened.
IComputeTraceSink * make_csv_compute_trace_sink(const std::string & path);
IIoTraceSink * make_csv_io_trace_sink(const std::string & path);

} // namespace bmoe
