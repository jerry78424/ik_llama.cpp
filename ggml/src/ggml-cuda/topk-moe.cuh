#include "common.cuh"

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               top_k,
                           ggml_tensor *               bias = nullptr,
                           bool                        sqrt_softplus = false);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights);

bool ggml_cuda_should_use_topk_moe_unary(const ggml_tensor * activation, const ggml_tensor * weights);
