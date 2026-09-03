#include "indexer_topk.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "convert.cuh"
#include "argsort.cuh"

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    for (int i = col; i < ncols; i += blockDim.x) {
        float sum = (float)this_m[i];
        auto this_kq = kq + blockIdx.x * step;
        for (int head = 0; head < nhead; ++head) {
            float relu = (float)this_kq[i];
            relu = relu > 0.0f ? relu : 0.0f;
            sum += relu * this_w[head];
            this_kq += ncols;
        }
        dst[ncols*row + i] = sum;
    }
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows_2(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = blockIdx.y*blockDim.x + threadIdx.x;
    if (col >= ncols) {
        return;
    }

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    float sum = (float)this_m[col];
    auto this_kq = kq + row * step;
    for (int head = 0; head < nhead; ++head) {
        float relu = (float)this_kq[col];
        relu = relu > 0.0f ? relu : 0.0f;
        sum += relu * this_w[head];
        this_kq += ncols;
    }
    dst[ncols*row + col] = sum;
}

static __global__ void k_copy_topk(const int * __restrict__ sorted, int * dst, const int ncols, const int n_top_k) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    sorted += int64_t(ncols)*row;
    dst    += int64_t(n_top_k)*row;
    for (int i = col; i < n_top_k; i += blockDim.x) {
        dst[i] = sorted[i];
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#    define IK_CUB_AVAILABLE 1
#    ifdef _WIN32
#        ifndef WIN32_LEAN_AND_MEAN
#            define WIN32_LEAN_AND_MEAN
#        endif
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#    endif
#    include <cub/cub.cuh>
using namespace cub;
#endif

// Exact top-k selection replacing the full per-row argsort in ggml_cuda_op_indexer_topk.
//
// The scores are m[i] + sum_head(relu(kq)*w[head]) with m drawn from the kq mask
// (0.0 valid / -INF masked) and w learned weights that may be negative, so the
// value domain is MIXED-SIGN. The descending float->uint key uses the monotonic
// transform empirically matched to CUB DeviceRadixSort::SortPairsDescending
// (verified 2026-09-02 with a standalone CUB probe on RTX 5090/sm_120):
//     key = (u & 0x80000000) ? ~u : (u | 0x80000000)
// so that -inf < ... < -0.0 < +0.0 < ... < +inf. Earlier attempts used
// "(u & 0x80000000) ? u : ~u" and "(u & 0x80000000) ? u : (~u ^ 0x80000000)",
// BOTH of which scramble the order (raw negative bits rank -inf above everything
// and reverse the positives); the first caused the 2026-09-02 output regression
// and the second was caught by the build gate (logits differed from baseline).
//
// The top-k set is extracted exactly:
//   1. per-row 256-bucket histogram of key >> 24 (bucket ascending = score ascending)
//   2. per-row boundary bucket b* scanned from the HIGHEST bucket down (the bucket
//      containing the k-th largest key), segment size = buckets > b* plus all of b*
//      (>= n_top_k by construction)
//   3. survivors (bucket >= b*) compacted per row; composite uint64 key
//      (key << 32) | (0xFFFFFFFF - index) so ties break by ascending index,
//      matching the full stable radix sort's tie order deterministically
//   4. DeviceSegmentedRadixSort::SortPairsDescending (capture-safe) over
//      survivors; first n_top_k of each segment copied to dst
// Downstream (ggml_indexer_mask, ggml_get_rows_ext) consumes the index SET only,
// so set-exactness is sufficient; the index tie-break makes it bit-exact vs the
// full sort and deterministic across runs.

__device__ __forceinline__ uint32_t indexer_topk_key(float f) {
    // Monotonic float -> uint transform for DESCENDING order, empirically matched
    // against CUB DeviceRadixSort::SortPairsDescending (see tools note 2026-09-02):
    //     -inf < ... < -0.0 < +0.0 < ... < +inf, ties by ascending index.
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}

static __global__ void k_topk_hist(const float * __restrict__ score,
        unsigned int * __restrict__ hist, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    __shared__ unsigned int s_hist[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();
    const float * s = score + (int64_t) row * ncols;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        atomicAdd(&s_hist[indexer_topk_key(s[i]) >> 24], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[(int64_t) row * 256 + i] = s_hist[i];
}

static __global__ void k_topk_boundary(const unsigned int * __restrict__ hist,
        unsigned int * __restrict__ seg_size, unsigned int * __restrict__ seg_pos,
        unsigned char * __restrict__ b_star, const int nrows, const int n_top_k) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const unsigned int * h = hist + (int64_t) row * 256;
    unsigned int c_less = 0;
    int b = 0;
    // Buckets are ordered ascending with score (key = monotonic float transform),
    // so scan from the HIGHEST bucket down to find the bucket holding the
    // k-th LARGEST key. c_less = count of keys in buckets ABOVE b.
    for (int i = 255; i >= 0; --i) {
        if (c_less + h[i] >= (unsigned int) n_top_k) { b = i; break; }
        c_less += h[i];
    }
    b_star[row]  = (unsigned char) b;
    seg_size[row] = c_less + h[b];
    seg_pos[row]  = 0;
}

static __global__ void k_topk_prefix(const unsigned int * __restrict__ seg_size,
        int * __restrict__ seg_offset, const int nrows) {
    constexpr int block = 256;
    __shared__ unsigned int s_sums[block];
    __shared__ unsigned int s_scan[block];
    // Each thread owns a CONTIGUOUS chunk of rows. A strided assignment here is
    // WRONG for nrows > 256: thread t's running prefix would then only cover its
    // own strided rows, missing the other threads' rows in between, so seg_offset
    // collides across rows and the compaction overwrites survivors (duplicate
    // indices in dst). nrows reaches ~1725 in the fast prefill path.
    const int chunk = (nrows + block - 1) / block;
    const int start = threadIdx.x * chunk;
    const int end = min(start + chunk, nrows);

    unsigned int local = 0;
    for (int i = start; i < end; ++i) local += seg_size[i];
    s_sums[threadIdx.x] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int acc = 0;
        for (int i = 0; i < block; ++i) { s_scan[i] = acc; acc += s_sums[i]; }
        s_sums[block - 1] = acc;  // total survivors
    }
    __syncthreads();

    unsigned int acc = s_scan[threadIdx.x];
    for (int i = start; i < end; ++i) {
        seg_offset[i] = (int) acc;
        acc += seg_size[i];
    }
    if (threadIdx.x == 0) seg_offset[nrows] = (int) s_sums[block - 1];
}

static __global__ void k_topk_compact(const float * __restrict__ score,
        const unsigned char * __restrict__ b_star, unsigned int * __restrict__ seg_pos,
        const int * __restrict__ seg_offset, unsigned long long * __restrict__ surv_keys,
        int * __restrict__ surv_vals, const int ncols, const int nrows) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const unsigned int b = b_star[row];
    const float * s = score + (int64_t) row * ncols;
    unsigned int * pos = seg_pos + row;
    const int base = seg_offset[row];
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        const uint32_t key = indexer_topk_key(s[i]);
        if ((key >> 24) >= b) {
            const uint32_t p = atomicAdd(pos, 1u);
            const unsigned long long k64 = ((unsigned long long) key << 32) | (0xFFFFFFFFu - (uint32_t) i);
            surv_keys[base + p] = k64;
            surv_vals[base + p] = i;
        }
    }
}

static __global__ void k_topk_copy_seg(const int * __restrict__ sorted_vals,
        const int * __restrict__ seg_offset, int * __restrict__ dst,
        const int n_top_k, const int nrows) {
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int * src = sorted_vals + seg_offset[row];
    int * d = dst + (int64_t) row * n_top_k;
    for (int i = threadIdx.x; i < n_top_k; i += blockDim.x) d[i] = src[i];
}

static __global__ void k_topk_verify(const int * __restrict__ full, const int * __restrict__ sel,
        const int n_top_k, const int nrows, int * __restrict__ mismatch) {
    const int row = blockIdx.x;
    extern __shared__ int s_sel[];
    const int * fs = full + (int64_t) row * n_top_k;
    const int * ss = sel  + (int64_t) row * n_top_k;
    for (int i = threadIdx.x; i < n_top_k; i += blockDim.x) s_sel[i] = ss[i];
    __syncthreads();
    int local = 0;
    for (int j = threadIdx.x; j < n_top_k; j += blockDim.x) {
        const int v = fs[j];
        bool found = false;
        for (int i = 0; i < n_top_k; ++i) {
            if (s_sel[i] == v) { found = true; break; }
        }
        if (!found) ++local;
    }
    if (local) atomicAdd(mismatch, local);
}

static void indexer_topk_select(ggml_backend_cuda_context & ctx, const float * score,
        int * dst, const int ncols, const int nrows, const int n_top_k, cudaStream_t stream) {
#ifdef IK_CUB_AVAILABLE
    ggml_cuda_pool_alloc<unsigned int> hist(ctx.pool(), (size_t) nrows * 256);
    ggml_cuda_pool_alloc<unsigned int> seg_size(ctx.pool(), (size_t) nrows);
    ggml_cuda_pool_alloc<unsigned int> seg_pos(ctx.pool(), (size_t) nrows);
    ggml_cuda_pool_alloc<unsigned char> b_star(ctx.pool(), (size_t) nrows);
    ggml_cuda_pool_alloc<int> seg_offset(ctx.pool(), (size_t) nrows + 1);
    ggml_cuda_pool_alloc<unsigned long long> surv_keys(ctx.pool(), (size_t) ncols * nrows);
    ggml_cuda_pool_alloc<int> surv_vals(ctx.pool(), (size_t) ncols * nrows);
    // Separate output buffers for the segmented sort: the CUB call is NOT allowed to alias
    // input/output here because for an odd number of passes (11 for 64-bit keys) pass 1 reads
    // AND writes the same buffer, which is only safe for single-tile segments. During decode the
    // survivors can far exceed a single tile (boundary bucket heavily populated), so in-place is
    // undefined behavior -> IMA. Distinct output buffers make every pass read/write different
    // buffers (surv -> sorted -> tmp -> sorted), safe for any segment size.
    ggml_cuda_pool_alloc<unsigned long long> sorted_keys(ctx.pool(), (size_t) ncols * nrows);
    ggml_cuda_pool_alloc<int> sorted_vals(ctx.pool(), (size_t) ncols * nrows);

    constexpr int k_block = 256;
    k_topk_hist<<<nrows, k_block, 0, stream>>>(score, hist.get(), ncols, nrows);
    CUDA_CHECK(cudaGetLastError());
    k_topk_boundary<<<nrows, k_block, 0, stream>>>(hist.get(), seg_size.get(), seg_pos.get(), b_star.get(), nrows, n_top_k);
    CUDA_CHECK(cudaGetLastError());
    k_topk_prefix<<<1, k_block, 0, stream>>>(seg_size.get(), seg_offset.get(), nrows);
    CUDA_CHECK(cudaGetLastError());
    k_topk_compact<<<nrows, k_block, 0, stream>>>(score, b_star.get(), seg_pos.get(), seg_offset.get(),
            surv_keys.get(), surv_vals.get(), ncols, nrows);
    CUDA_CHECK(cudaGetLastError());

    const int64_t num_items = (int64_t) ncols * nrows;  // upper bound; segments delimit the real data
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_storage_bytes,
            surv_keys.get(), sorted_keys.get(), surv_vals.get(), sorted_vals.get(),
            (int) num_items, nrows, seg_offset.get(), seg_offset.get() + 1, 0, sizeof(unsigned long long) * 8, stream));
    ggml_cuda_pool_alloc<uint8_t> temp_storage(ctx.pool(), temp_storage_bytes);
    CUDA_CHECK(DeviceSegmentedRadixSort::SortPairsDescending(temp_storage.get(), temp_storage_bytes,
            surv_keys.get(), sorted_keys.get(), surv_vals.get(), sorted_vals.get(),
            (int) num_items, nrows, seg_offset.get(), seg_offset.get() + 1, 0, sizeof(unsigned long long) * 8, stream));
    CUDA_CHECK(cudaGetLastError());

    k_topk_copy_seg<<<nrows, k_block, 0, stream>>>(sorted_vals.get(), seg_offset.get(), dst, n_top_k, nrows);
    CUDA_CHECK(cudaGetLastError());

    static const bool verify = []() {
        const char * e = getenv("IK_INDEXER_VERIFY");
        return e != nullptr && e[0] != '\0';
    }();
    if (verify) {
        // Only meaningful outside stream capture (the full argsort path is chosen
        // by the same capture state, but the host sync below is illegal in capture).
        cudaStreamCaptureStatus cap;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap));
        if (cap == cudaStreamCaptureStatusNone && n_top_k <= 1024) {
            ggml_cuda_pool_alloc<int> sorted(ctx.pool(), (size_t) ncols * nrows);
            argsort_f32_i32_cuda_cub(ctx.pool(), score, sorted.get(), ncols, nrows, GGML_SORT_ORDER_DESC, stream);
            ggml_cuda_pool_alloc<int> mismatch(ctx.pool(), 1);
            CUDA_CHECK(cudaMemsetAsync(mismatch.get(), 0, sizeof(int), stream));
            const int smem = n_top_k * sizeof(int);
            k_topk_verify<<<nrows, k_block, smem, stream>>>(sorted.get(), dst, n_top_k, nrows, mismatch.get());
            CUDA_CHECK(cudaGetLastError());
            int h = 0;
            CUDA_CHECK(cudaMemcpyAsync(&h, mismatch.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, "%s: IK_INDEXER_VERIFY: ncols=%d nrows=%d n_top_k=%d top-k set %s\n",
                    __func__, ncols, nrows, n_top_k, h == 0 ? "MATCH" : "MISMATCH");
        }
    }
#else
    // No CUB: fall back to the full per-row argsort.
    ggml_cuda_pool_alloc<int> sorted(ctx.pool(), (size_t) ncols * nrows);
    argsort_f32_i32_cuda_cub(ctx.pool(), score, sorted.get(), ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    k_copy_topk<<<nrows, k_block, 0, stream>>>(sorted.get(), dst, ncols, n_top_k);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void ggml_cuda_op_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto op = ggml_unary_op(dst->op_params[0]);
    GGML_ASSERT(op == GGML_UNARY_OP_RELU);
    auto k = dst->src[0];
    auto q = dst->src[1];
    auto w = dst->src[2];
    auto m = dst->src[3];
    int n_top_k = dst->ne[0];
    int n_kv    = k->ne[1];
    int n_head  = q->ne[1];
    //if (k->type != GGML_TYPE_F16 && !ggml_is_quantized(k->type)) printf("%s: K is %s?\n", __func__, ggml_type_name(k->type));
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_F32 || ggml_is_quantized(k->type));
    GGML_ASSERT(k->ne[2] == 1 || k->ne[3] == 1);
    GGML_ASSERT(k->ne[1] > n_top_k);
    GGML_ASSERT(k->ne[1] == m->ne[0]);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    GGML_ASSERT(q->ne[2] == m->ne[1]);
    GGML_ASSERT(q->ne[1] == w->ne[0]);
    GGML_ASSERT(q->ne[2] == w->ne[1]);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F32 || m->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(w));

    constexpr int k_block_size = 256;

    if ((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && q->type == GGML_TYPE_F32) {
        constexpr size_t k_max_buf_size = 1 << 28;
        size_t per_row = size_t(n_kv)*(q->ne[1]*sizeof(half) + sizeof(int) + sizeof(float)) + q->ne[0]*q->ne[1]*sizeof(half);
        int max_rows = (k_max_buf_size + per_row - 1)/per_row;
        max_rows = std::min<int>(max_rows, q->ne[2]);
        int nstep = (q->ne[2] + max_rows - 1)/max_rows;

        ggml_cuda_pool_alloc<half>  kq(ctx.pool(), int64_t(n_kv)*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<half>  q_f16(ctx.pool(), q->ne[0]*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<half>  k_f16(ctx.pool());

        auto k_data = (const half *)k->data;
        if (k->type == GGML_TYPE_Q8_0) {
            k_f16.alloc(k->ne[0]*k->ne[1]);
            auto to_fp16_cuda = ggml_get_to_fp16_cuda(k->type);
            GGML_ASSERT(to_fp16_cuda);
            to_fp16_cuda(k->data, k_f16.get(), k->ne[0]*k->ne[1], 1, ctx.stream());
            CUDA_CHECK(cudaGetLastError());
            k_data = k_f16.get();
        }

        auto to_fp16_cuda = ggml_get_to_fp16_cuda(q->type);
        GGML_ASSERT(to_fp16_cuda);

        const half alpha = 1.0f;
        const half beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));

        for (int istep = 0; istep < nstep; ++istep) {
            int first_row = max_rows*istep;
            if (first_row >= int(q->ne[2])) break;
            int last_row  = std::min(first_row + max_rows, int(q->ne[2]));
            int nrows     = last_row - first_row;

            to_fp16_cuda((const float *)q->data + q->ne[0]*q->ne[1]*first_row, q_f16.get(), q->ne[0]*q->ne[1]*nrows, 1, ctx.stream());
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*nrows, q->ne[0],
                    &alpha, k_data,       CUDA_R_16F, k->ne[0],
                            q_f16.get(),  CUDA_R_16F, q->ne[0],
                    &beta,  kq.get(),     CUDA_R_16F, k->ne[1],
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            int nblocks = (k->ne[1] + k_block_size - 1)/k_block_size;
            dim3 grid(nrows, nblocks, 1);
            if (m->type == GGML_TYPE_F32) {
                k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                        (const float *)w->data + first_row*q->ne[1],
                        (const float *)((const char *)m->data + first_row*m->nb[1]),
                        score.get(), k->ne[1], q->ne[1], m->nb[1]);
            } else {
                k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                        (const float *)w->data + first_row*q->ne[1],
                        (const half  *)((const char *)m->data + first_row*m->nb[1]),
                        score.get(), k->ne[1], q->ne[1], m->nb[1]);
            }
            CUDA_CHECK(cudaGetLastError());

            indexer_topk_select(ctx, score.get(),
                    (int *)((char *)dst->data + first_row*dst->nb[1]), k->ne[1], nrows, dst->ne[0], ctx.stream());
            CUDA_CHECK(cudaGetLastError());
        }

        return;

    }

    constexpr int64_t k_max_work_buffer_elements = 1 << 26;

    int max_rows = k_max_work_buffer_elements / n_kv / n_head;
    if (max_rows < 1) max_rows = 1;
    if (max_rows > q->ne[2]) max_rows = q->ne[2];

    int nstep = (q->ne[2] + max_rows - 1)/max_rows;

    ggml_cuda_pool_alloc<float> kq(ctx.pool(), int64_t(n_kv)*max_rows*n_head);
    ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool());
    ggml_cuda_pool_alloc<char>  q_converted(ctx.pool());
    const float * k_data = nullptr;
    int k_ld = k->ne[0];
    auto q_padded = GGML_PAD(q->ne[0], MATRIX_ROW_PADDING);
    if (ggml_is_quantized(k->type)) {
        auto nbytes_q = (size_t)(q_padded/QK8_1) * ((size_t) q->ne[1] * max_rows) * sizeof(block_q8_1);
        nbytes_q += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        q_converted.alloc(nbytes_q);
    } else if (k->type == GGML_TYPE_F32) {
        k_data = (const float *) k->data;
        k_ld   = k->nb[1]/sizeof(float);
    } else {
        k_f32.alloc(k->ne[0]*k->ne[1]);
        ggml_get_to_fp32_cuda(k->type)(k->data, k_f32.get(), k->ne[1]*k->ne[0], 1, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        k_data = k_f32.get();
    }

    for (int istep = 0; istep < nstep; ++istep) {
        int first = istep*max_rows;
        int last  = std::min(first + max_rows, int(q->ne[2]));
        int nrows = last - first;
        auto q_data = (const char *)q->data + istep*max_rows*q->nb[2];
        auto m_data = (const char *)m->data + istep*max_rows*m->nb[1];
        auto w_data = (const float *)w->data + first*q->ne[1];
        if (ggml_is_quantized(k->type)) {
            quantize_mmq_q8_1_cuda((const float *)q_data, q_converted.get(), q->ne[0], q->ne[1]*nrows, 1, q_padded, k->type, ctx.stream());
            CUDA_CHECK(cudaGetLastError());
            mmq_args args{(const char *)k->data, q_converted.get(), kq.get(),
                k->ne[0], k->ne[1], int64_t(k->nb[1]),
                q_padded, q->ne[1]*nrows, q->ne[1]*nrows, k->ne[1]};
            ggml_cuda_op_mul_mat_q(ctx, k->type, args);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // I wonder if it makes sense to use CUBLAS. If we did simple dot products we could fuse the
            // relu, mul, sum_rows all in one kernel, avoiding the k*q intermediate result.
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
            CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*nrows, q->ne[0],
                    &alpha,     k_data,                k_ld,
                       (const float *)q_data, q->ne[0],
                    &beta,      kq.get(),     k->ne[1]));
        }
        if (m->type == GGML_TYPE_F32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const float *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const half  *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());

        indexer_topk_select(ctx, score.get(),
                (int *)((char *)dst->data + first*dst->nb[1]), k->ne[1], nrows, dst->ne[0], ctx.stream());
        CUDA_CHECK(cudaGetLastError());
    }

}

template <typename mask_t>
static __global__ void k_indexer_mask(int ne0, int ne1, int ne2, int ntopk, int ne11,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * mask, const int * idx, mask_t * dst) {
    int i1 = blockIdx.x;
    int i3 = i1 / (ne1*ne2); i1 -= i3*ne1*ne2;
    int i2 = i1 / (ne1);     i1 -= i2*ne1;

    auto m = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    auto i = (const int *)((const char *)idx + i1*nb11 + i2*nb12 + i3*nb13);
    auto d = (mask_t *)((char *)dst + i1*nb1 + i2*nb2 + i3*nb3);

    mask_t inf, zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        inf  = __float2half(-INFINITY);
        zero = __float2half(0.0f);
    } else {
        inf  = -INFINITY;
        zero = 0.0f;
    }

    if (i1 < ne11) {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = inf;
        __syncthreads();
        for (int j = threadIdx.x; j < ntopk; j += blockDim.x) d[i[j]] = zero;
        __syncthreads();
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] += m[j];
    } else {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = m[j];
    }

}

void ggml_cuda_op_indexer_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto mask = dst->src[0];
    auto topk = dst->src[1];
    GGML_ASSERT(mask->ne[0] >= topk->ne[0]);
    GGML_ASSERT(mask->ne[1] >= topk->ne[1] && mask->ne[2] == topk->ne[2] && mask->ne[3] == topk->ne[3]);
    GGML_ASSERT(ggml_are_same_shape(mask, dst));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(topk->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == dst->type);

    int nrows = ggml_nrows(dst);

    if (dst->type == GGML_TYPE_F16) {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const half *)mask->data, (const int *)topk->data, (half *)dst->data);
    } else {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const float *)mask->data, (const int *)topk->data, (float *)dst->data);
    }

}

template <typename mask_t>
static __global__ void k_mask_to_index(int ne00, [[maybe_unused]] int ne0,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * __restrict__ mask, int * __restrict__ idx) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;

    mask_t zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        zero = __float2half(0.0f);
    } else {
        zero = 0.0f;
    }
    __shared__ int counts[WARP_SIZE];
    const mask_t * mask_r = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    int * idx_r = (int *)((char *)idx + i1*nb1 + i2*nb2 + i3*nb3);

    for (int j = threadIdx.x; j < ne0; j += WARP_SIZE) {
        idx_r[j] = -1;
    }

    int nOn = 0;
    for (int j = threadIdx.x; j < ne00; j += WARP_SIZE) {
        nOn += (mask_r[j] == zero ? 1 : 0);
    }
    counts[threadIdx.x] = nOn;
    __syncthreads();
    int cum[WARP_SIZE];
    int start = 0;
    for (int i = 0; i < WARP_SIZE; ++i) {
        cum[i] = start;
        start += counts[i];
    }
    start = cum[threadIdx.x];
    for (int j = threadIdx.x; j < ne00; j += WARP_SIZE) {
        if (mask_r[j] == zero) idx_r[start++] = j;
    }
}

void ggml_cuda_op_mask_to_index(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->ne[1] == dst->ne[1] && src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3]);
    GGML_ASSERT(src->ne[0] >= dst->ne[0]);

    dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
    if (src->type == GGML_TYPE_F16) {
        k_mask_to_index<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const half *)src->data, (int *)dst->data);
    } else {
        k_mask_to_index<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const float *)src->data, (int *)dst->data);
    }

}
