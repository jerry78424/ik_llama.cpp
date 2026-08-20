#include "dsv4_trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"

namespace dsv4_trace {

static bool enabled() {
    return std::getenv("IK_DSV4_TRACE") != nullptr;
}

static uint64_t seq_counter = 0;

void emit(const char * tag, const char * fmt, ...) {
    if (!enabled()) {
        return;
    }
    const uint64_t seq = ++seq_counter;
    const uint64_t t_us = (uint64_t) ggml_time_us();
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // The fmt fragment is a set of "key":value pairs; wrap it into a JSON object.
    printf("DSV4TRACE {\"seq\":%llu,\"t_us\":%llu,\"tag\":\"%s\",%s}\n",
            (unsigned long long) seq, (unsigned long long) t_us, tag, buf);
    fflush(stdout);
}

void emit_set_input(const ggml_tensor * tensor, const void * values, size_t n, size_t esize) {
    if (!enabled() || tensor == nullptr || tensor->name == nullptr || values == nullptr) {
        return;
    }
    char vals[1024];
    vals[0] = '\0';
    size_t off = 0;
    const size_t nshow = n < 6 ? n : 6;
    for (size_t i = 0; i < nshow; ++i) {
        long long v = 0;
        if (esize == sizeof(int32_t)) {
            v = (long long) ((const int32_t *) values)[i];
        } else {
            v = (long long) ((const int64_t *) values)[i];
        }
        int w = snprintf(vals + off, sizeof(vals) - off, "%s%lld", i == 0 ? "" : ",", v);
        if (w < 0) break;
        off += (size_t) w;
    }
    emit("set_input", "\"name\":\"%s\",\"ne0\":%lld,\"n\":%zu,\"vals\":[%s]", tensor->name,
            (long long) tensor->ne[0], n, vals);
}

void emit_tensor(const char * tag, const ggml_tensor * t, int64_t row, int64_t nelem) {
    if (!enabled() || t == nullptr || t->buffer == nullptr) {
        return;
    }
    // Hash every row of the tensor.
    const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    const int64_t nrows = t->ne[1];
    uint64_t h = 1469598103934665603ULL;
    std::vector<uint8_t> buf(row_bytes);
    for (int64_t r = 0; r < nrows; ++r) {
        ggml_backend_tensor_get((ggml_tensor *) t, buf.data(), (size_t) r * row_bytes, row_bytes);
        const uint8_t * p = buf.data();
        for (size_t i = 0; i < row_bytes; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }

    // If a specific row is requested, dump its leading elements.
    char rowdump[2048];
    rowdump[0] = '\0';
    if (row >= 0 && row < nrows) {
        std::vector<float> fbuf(row_bytes/sizeof(float));
        ggml_backend_tensor_get((ggml_tensor *) t, fbuf.data(), (size_t) row * row_bytes, row_bytes);
        size_t off = 0;
        const size_t nshow = nelem < 8 ? nelem : 8;
        for (size_t e = 0; e < nshow; ++e) {
            int w = snprintf(rowdump + off, sizeof(rowdump) - off, "%s%.8g", e == 0 ? "" : ",", fbuf[e]);
            if (w < 0) break;
            off += (size_t) w;
        }
    }

    emit(tag, "\"name\":\"%s\",\"type\":\"%s\",\"ne\":%lldx%lld,\"row\":%lld,\"hash\":%016llx,\"elems\":[%s]",
            t->name, ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1],
            (long long) row, (unsigned long long) h, rowdump);
}

}