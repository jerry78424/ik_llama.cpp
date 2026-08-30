#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params, const void * model);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// DSV4 device-side kq_mask fill: writes mask[i][j] = (j < n_visible[i]) ? 0 : -INF
// directly into the input tensor's CUDA buffer. Returns false when the tensor is
// not on a regular CUDA buffer (caller must fall back to the host path).
GGML_API GGML_CALL bool ggml_backend_cuda_dsv4_fill_mask(struct ggml_tensor * mask, const int32_t * n_visible, int64_t n_tokens);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_API GGML_CALL void ggml_backend_cuda_invalidate_graphs(const void * model);
#ifdef  __cplusplus
}
#endif
