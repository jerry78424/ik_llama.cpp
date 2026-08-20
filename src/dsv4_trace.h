#pragma once

// Centralized DSV4 tracing emitter.
//
// All DSV4 instrumentation funnels through this single module. It emits one
// self-contained JSON object per line to stdout, prefixed with "DSV4TRACE " so an
// external analyzer (tools/dsv4_harness/analyze.py) can separate trace events from
// ordinary server log lines. Each event carries a monotonically increasing "seq"
// and a "tag" so the analyzer can align batch-vs-seq runs and attribute writes to
// the correct decode.
//
// Everything is gated behind the IK_DSV4_TRACE environment variable; when unset the
// emitters are no-ops and the production build is unaffected.

#include <cstdarg>
#include <cstdint>

#include "ggml.h"

namespace dsv4_trace {

    // Emit a generic event. fmt is a pre-formatted JSON fragment (a set of
    // "key":value pairs, comma-separated, no braces). The emitter wraps it as:
    //   DSV4TRACE {"seq":N,"t_us":<us>,"tag":"<tag>",<fmt>}
    // No-op unless IK_DSV4_TRACE is set.
    void emit(const char * tag, const char * fmt, ...);

    // Emit a set_input event for the raw_k_write index tensors. Builds the "vals"
    // array from the given element array (int32/int64). No-op unless IK_DSV4_TRACE is set.
    void emit_set_input(const ggml_tensor * tensor, const void * values, size_t n, size_t esize);

    // Emit a tensor readback event: hashes the given tensor's rows and, when a
    // specific row is requested, dumps that row's leading elements. No-op unless
    // IK_DSV4_TRACE is set.
    void emit_tensor(const char * tag, const ggml_tensor * t, int64_t row, int64_t nelem);

}