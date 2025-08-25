#include <type_traits>
#include <variant>
#include <cuda_bf16.h>
#include <stdexcept>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
using namespace nvcuda;

#define FULL_MASK 0xFFFFFFFF

template <typename T>
struct native_pack;

template <>
struct native_pack<float> {
  using T2 = float2;
};

# if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)

template <>
struct native_pack<nv_bfloat16> {
  using T2 = nv_bfloat162;
};


template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask >>= 1) val += __shfl_xor_sync(FULL_MASK, val, mask, 32);
  return val;
}

typedef struct bf164 {
    nv_bfloat16 x, y, z, w;
} bf164;

static __inline__ __host__ __device__ float4 make_float4(nv_bfloat16 x, nv_bfloat16 y, nv_bfloat16 z, nv_bfloat16 w)
{
  float4 t; t.x = __bfloat162float(x); t.y = __bfloat162float(y); t.z = __bfloat162float(z); t.w = __bfloat162float(w); return t;
}

__inline__ __device__ int4 load_vector(const int* ptr) {
  return *(const int4*)ptr;
}

__inline__ __device__ float4 load_vector(const float* ptr) {
  return *(const float4*)ptr;
}

__inline__ __device__ float4 load_vector(const nv_bfloat16* ptr) {
  bf164 tmp;
  *(float2*)(&tmp) = *(const float2*)ptr;
  return make_float4(tmp.x, tmp.y, tmp.z, tmp.w);
}

__inline__ __device__ void store_vector(int* ptr, int4 x) {
  *(int4*)ptr = x;
}

__inline__ __device__ void store_vector(float* ptr, float4 x) {
  *(float4*)ptr = x;
}

__inline__ __device__ void store_vector(nv_bfloat16* ptr, float4 x) {
  bf164 tmp;
  tmp.x = __float2bfloat16(x.x), tmp.y = __float2bfloat16(x.y), tmp.z = __float2bfloat16(x.z), tmp.w = __float2bfloat16(x.w);
  *(float2*)ptr = *(float2*)(&tmp);
}


template <typename T>
__inline__ __device__ T ConvertTo(nv_bfloat16 x) {
  return static_cast<T>(x);
}


template <>
__inline__ __device__ float ConvertTo<float>(nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <>
__inline__ __device__ nv_bfloat16 ConvertTo<nv_bfloat16>(nv_bfloat16 x) {
  return x;
}

template<bool has_padding_idx>
__global__ void glu_forward_naive_kernel(const int* index, const __nv_bfloat16* weight, const __nv_bfloat16* p_input, __nv_bfloat16* output,
                                         const int vocab_size, const int per_layer_vocab_size, const int shift,
                                         const int reduce_dim, const int reduce_dim_align, const int embedding_dim, const int group_size, const int padding_idx) {
  extern __shared__ char smem[];
  float* s_p_input = (float*)smem;
  for (int tid = threadIdx.x; tid < embedding_dim * group_size; tid += blockDim.x) {
    s_p_input[tid] = __bfloat162float(p_input[blockIdx.x * embedding_dim * group_size + tid]);
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < reduce_dim; tid += blockDim.x) {
    int index_id = index[blockIdx.x * reduce_dim + tid];
    int group_id = 0;
    if(group_size > 1){
      group_id = index_id / per_layer_vocab_size;
    }
    float v = 0.0f;
    if (!has_padding_idx || index_id != padding_idx) {
      index_id = (index_id % per_layer_vocab_size + shift) % vocab_size;
      for (int i = 0; i < embedding_dim; i++) {
        float v1 = __bfloat162float(weight[(int64_t)index_id * embedding_dim + i]);
        float v2 = s_p_input[group_id * embedding_dim + i];
        v += v1 * v2;
      }
    }
    output[blockIdx.x * reduce_dim + tid] = __float2bfloat16(v);
  }
}

template<bool has_padding_idx>
__global__ void glu_forward_v4_kernel(const int* index, const __nv_bfloat16* weight, const __nv_bfloat16* p_input, __nv_bfloat16* output,
                                      const int vocab_size, const int per_layer_vocab_size, const int shift,
                                      const int reduce_dim, const int reduce_dim_align, const int embedding_dim, const int group_size, const int padding_idx) {
  extern __shared__ char smem[];
  float* s_p_input = (float*)smem;
  for (int tid = threadIdx.x; tid < embedding_dim * group_size; tid += blockDim.x) {
    s_p_input[tid] = __bfloat162float(p_input[blockIdx.x * embedding_dim * group_size + tid]);
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < reduce_dim; tid += blockDim.x) {
    int index_id = index[blockIdx.x * reduce_dim + tid];
    int group_id = 0;
    if(group_size > 1){
      group_id = index_id / per_layer_vocab_size;
    }
    float v = 0.0f;
    if (!has_padding_idx || index_id != padding_idx) {
      index_id = (index_id % per_layer_vocab_size + shift) % vocab_size;
      for (int i = 0; i < embedding_dim; i += 4) {
        float4 v1 = load_vector(weight + (int64_t)index_id * embedding_dim + i);
        float4 v2 = load_vector(s_p_input + group_id * embedding_dim + i);
        v += v1.x * v2.x;
        v += v1.y * v2.y;
        v += v1.z * v2.z;
        v += v1.w * v2.w;
      }
    }
    output[blockIdx.x * reduce_dim + tid] = __float2bfloat16(v);
  }
}

static bool sufficient(int smem_size) {
  int device_idx;
  cudaError_t result = cudaGetDevice(&device_idx);

  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() failed");
  }

  int sharedMemPerBlock;
  result = cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx);

  if (result != cudaSuccess) {
    throw std::runtime_error("cudaDeviceGetAttribute() failed");
  }

  return sharedMemPerBlock >= smem_size;
}

torch::Tensor GluForward(const torch::Tensor& indices, const torch::Tensor& weight, const torch::Tensor& p_input,
                         int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx) {
  TORCH_INTERNAL_ASSERT(indices.scalar_type() == at::ScalarType::Int);
  TORCH_INTERNAL_ASSERT(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(p_input.scalar_type() == at::ScalarType::BFloat16);
  std::vector<int64_t> indices_shape;
  for (int i = 0; i < indices.dim(); ++i) {
    indices_shape.push_back(indices.size(i));
  }
  std::vector<int64_t> weight_shape;
  for (int i = 0; i < weight.dim(); ++i) {
    weight_shape.push_back(weight.size(i));
  }
  const int batch_size = indices_shape[0];
  const int reduce_dim = indices_shape[1];
  const int reduce_dim_align = reduce_dim + 15 & -16;
  const int embedding_dim = weight_shape[1];
  std::vector<int64_t> output_shape = indices_shape;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  auto act_options = weight.options().requires_grad(false);
  torch::Tensor output = torch::empty(output_shape, act_options);
  const int shmem_size = sizeof(float) * embedding_dim * group_size;
  TORCH_INTERNAL_ASSERT(sufficient(shmem_size));
  if (embedding_dim % 4 == 0) {
    if (has_padding_idx) {
      cudaFuncSetAttribute(glu_forward_v4_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
      glu_forward_v4_kernel<true><<<batch_size, 128, shmem_size, stream>>>(
        indices.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(p_input.data<T>()), reinterpret_cast<__nv_bfloat16*>(output.data<T>()),
        vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, group_size, padding_idx);
    } else {
      cudaFuncSetAttribute(glu_forward_v4_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
      glu_forward_v4_kernel<false><<<batch_size, 128, shmem_size, stream>>>(
        indices.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(p_input.data<T>()), reinterpret_cast<__nv_bfloat16*>(output.data<T>()),
        vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, group_size, padding_idx);
    }
  } else {
    if (has_padding_idx) {
      cudaFuncSetAttribute(glu_forward_naive_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
      glu_forward_naive_kernel<true><<<batch_size, 128, shmem_size, stream>>>(
        indices.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(p_input.data<T>()), reinterpret_cast<__nv_bfloat16*>(output.data<T>()),
        vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, group_size, padding_idx);
    } else {
      cudaFuncSetAttribute(glu_forward_naive_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
      glu_forward_naive_kernel<false><<<batch_size, 128, shmem_size, stream>>>(
        indices.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(p_input.data<T>()), reinterpret_cast<__nv_bfloat16*>(output.data<T>()),
        vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, group_size, padding_idx);
    }
  }
  return output;
}

template<bool has_padding_idx>
__global__ void glu_backward_naive_kernel(const int* index, const __nv_bfloat16* weight, const __nv_bfloat16* p_input, const __nv_bfloat16* output_grad,
                                          float* weight_grad, __nv_bfloat16* p_input_grad, const int vocab_size, const int per_layer_vocab_size, const int shift,
                                          const int reduce_dim, const int reduce_dim_align, const int embedding_dim, const int group_size, const int padding_idx) {
  extern __shared__ char smem[];
  int* s_index = (int*)smem;
  float* s_output_grad = (float*)(smem + reduce_dim_align * 4);
  for (int tid = threadIdx.x; tid < reduce_dim; tid += blockDim.x) {
    s_index[tid] = index[blockIdx.x * reduce_dim + tid];
    s_output_grad[tid] = __bfloat162float(output_grad[blockIdx.x * reduce_dim + tid]);
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < embedding_dim; tid += blockDim.x) {
    float v2[4];
    for (int group_id = 0; group_id < group_size; group_id++) {
      v2[group_id] = __bfloat162float(p_input[(blockIdx.x * group_size + group_id) * embedding_dim + tid]);
    }
    float dv[4] = {0.0f};
    for (int i = 0; i < reduce_dim; i++) {
      int index_id = s_index[i];
      int group_id = 0;
      if(group_size > 1){
        group_id = index_id / per_layer_vocab_size;
      }
      if (has_padding_idx && index_id == padding_idx) {
        continue;
      }
      index_id = (index_id % per_layer_vocab_size + shift) % vocab_size;
      float v = s_output_grad[i];
      float v1 = __bfloat162float(weight[(int64_t)index_id * embedding_dim + tid]);
      atomicAdd(&weight_grad[index_id * embedding_dim + tid], v * v2[group_id]);
      dv[group_id] += v * v1;
    }
    for (int group_id = 0; group_id < group_size; group_id++) {
      p_input_grad[(blockIdx.x * group_size + group_id) * embedding_dim + tid] = __float2bfloat16(dv[group_id]);
    }
  }
}

torch::Tensor GluBackward(const torch::Tensor& indices, const torch::Tensor& weight, const torch::Tensor& p_input, const torch::Tensor& output_grad, torch::Tensor &weight_grad,
                          int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx) {
  TORCH_INTERNAL_ASSERT(indices.scalar_type() == at::ScalarType::Int);
  TORCH_INTERNAL_ASSERT(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(p_input.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(output_grad.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(weight_grad.scalar_type() == at::ScalarType::Float);
  std::vector<int64_t> indices_shape;
  for (int i = 0; i < indices.dim(); ++i) {
    indices_shape.push_back(indices.size(i));
  }
  std::vector<int64_t> p_input_shape;
  for (int i = 0; i < p_input.dim(); ++i) {
    p_input_shape.push_back(p_input.size(i));
  }
  std::vector<int64_t> weight_shape;
  for (int i = 0; i < weight.dim(); ++i) {
    weight_shape.push_back(weight.size(i));
  }
  const int batch_size = indices_shape[0];
  const int reduce_dim = indices_shape[1];
  const int reduce_dim_align = reduce_dim + 15 & -16;
  const int embedding_dim = weight_shape[1];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  auto act_options = p_input.options().requires_grad(false);
  torch::Tensor p_input_grad = torch::empty(p_input_shape, act_options);
  TORCH_INTERNAL_ASSERT(group_size <= 4);

  const int shmem_size = (sizeof(float) * group_size + sizeof(int)) * reduce_dim_align;
  TORCH_INTERNAL_ASSERT(sufficient(shmem_size));
  if (has_padding_idx) {
    cudaFuncSetAttribute(glu_backward_naive_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    glu_backward_naive_kernel<true><<<batch_size, 128, shmem_size, stream>>>(
      indices.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(p_input.data<T>()), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()),
      reinterpret_cast<float*>(weight_grad.data<float>()), reinterpret_cast<__nv_bfloat16*>(p_input_grad.data<T>()),
      vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, group_size, padding_idx);
  } else {
    cudaFuncSetAttribute(glu_backward_naive_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    glu_backward_naive_kernel<false><<<batch_size, 128, shmem_size, stream>>>(
      indices.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(p_input.data<T>()), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()),
      reinterpret_cast<float*>(weight_grad.data<float>()), reinterpret_cast<__nv_bfloat16*>(p_input_grad.data<T>()),
      vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, group_size, padding_idx);
  }

  return p_input_grad;
}

#endif
