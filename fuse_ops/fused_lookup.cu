
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

__device__ inline void __pipeline_memcpy_async_size4_with_zero_pad(void* smem, const void* gmem, int pred) {
  int s = __cvta_generic_to_shared(smem);
  asm volatile(
    "{.reg .pred p;\n\t"
    "setp.eq.u32 p, %0, 0;\n\t"
    "cp.async.ca.shared.global [%1], [%2], 4, p;}\n\t" : : "r"(pred), "r"(s), "l"(gmem));
}

__device__ inline void __pipeline_memcpy_async_size8_with_zero_pad(void* smem, const void* gmem, int pred) {
  int s = __cvta_generic_to_shared(smem);
  asm volatile(
    "{.reg .pred p;\n\t"
    "setp.eq.u32 p, %0, 0;\n\t"
    "cp.async.ca.shared.global [%1], [%2], 8, p;}\n\t" : : "r"(pred), "r"(s), "l"(gmem));
}

__device__ inline void __pipeline_memcpy_async_size16_with_zero_pad(void* smem, const void* gmem, int pred) {
  int s = __cvta_generic_to_shared(smem);
  asm volatile(
    "{.reg .pred p;\n\t"
    "setp.eq.u32 p, %0, 0;\n\t"
    "cp.async.cg.shared.global [%1], [%2], 16, p;}\n\t" : : "r"(pred), "r"(s), "l"(gmem));
}

template<bool has_padding_idx>
__global__ void lookup_reduce_forward_wmma_kernel(
    const int* index, const __nv_bfloat16* weight, const __nv_bfloat16* score, __nv_bfloat16* output,
    const int batch_size, const int vocab_size, const int per_layer_vocab_size, const int shift, const int reduce_dim, const int reduce_dim_align,
    const int embedding_dim, const int each_core_dim, const int group_size, const int padding_idx) {
  extern __shared__ char smem[];
  __nv_bfloat16* s_score = (__nv_bfloat16*)smem;
  int* s_org_index = (int*)(smem + reduce_dim_align * 2);
  int* s_index = (int*)(smem + reduce_dim_align * 6);
  __nv_bfloat16* s_table = (__nv_bfloat16*)(smem + reduce_dim_align * 10);
  __nv_bfloat16* s_masked_score = (__nv_bfloat16*)(smem + reduce_dim_align * 10 + 32 * each_core_dim * 2);
  float* result_buf = (float*)s_table;
  for (int tid = threadIdx.x; tid < reduce_dim_align; tid += blockDim.x) {
    ((float4*)s_masked_score)[tid]  = { 0, 0, 0, 0 };
  }
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  __syncthreads();

  wmma::fragment<wmma::matrix_a, 32, 8, 16, __nv_bfloat16, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, 32, 8, 16, __nv_bfloat16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  int each_core_offset = warp_id * 32 + (lane_id & 3) * 8;
  int offset = each_core_offset + blockIdx.y * each_core_dim;

  for (int tid = threadIdx.x; tid < reduce_dim; tid += blockDim.x) {
    int index_id = __ldg(&index[blockIdx.x * reduce_dim + tid]);
    if (has_padding_idx) {
      s_org_index[tid] = index_id;
    }
    int group_id =  index_id / per_layer_vocab_size;
    s_index[tid] = (index_id % per_layer_vocab_size + shift) % vocab_size;
    s_masked_score[tid * 8 + group_id] = __ldg(&score[(blockIdx.y * gridDim.x + blockIdx.x) * reduce_dim + tid]);
  }
  __syncthreads();

  int i = 0;
  int64_t index_id_0 = s_index[i + (lane_id >> 2)];
  int64_t index_id_1 = s_index[i + (lane_id >> 2) + 8];
  int org_index_id_0;
  int org_index_id_1;
  if (has_padding_idx) {
    org_index_id_0 = s_org_index[i + (lane_id >> 2)];
    org_index_id_1 = s_org_index[i + (lane_id >> 2) + 8];
  }
  auto cp_src_0 = weight + offset + index_id_0 * embedding_dim;
  auto cp_dst_0 = s_table + warp_id * 32 * 16 + lane_id * 8;
  auto cp_src_1 = weight + offset + index_id_1 * embedding_dim;
  auto cp_dst_1 = cp_dst_0 + 32 * 8;
  __pipeline_memcpy_async_size16_with_zero_pad(cp_dst_0, cp_src_0, each_core_offset < each_core_dim && i + (lane_id >> 2) < reduce_dim && (!has_padding_idx || org_index_id_0 != padding_idx));
  __pipeline_memcpy_async_size16_with_zero_pad(cp_dst_1, cp_src_1, each_core_offset < each_core_dim && i + (lane_id >> 2) < reduce_dim - 8 && (!has_padding_idx || org_index_id_1 != padding_idx));
  __pipeline_commit();

  #pragma unroll (1)
  for (i = 16; i < reduce_dim; i += 16) {
    index_id_0 = s_index[i + (lane_id >> 2)];
    index_id_1 = s_index[i + (lane_id >> 2) + 8];
    if (has_padding_idx) {
      org_index_id_0 = s_org_index[i + (lane_id >> 2)];
      org_index_id_1 = s_org_index[i + (lane_id >> 2) + 8];
    }
    cp_src_0 = weight + offset + index_id_0 * embedding_dim;
    cp_dst_0 = s_table + warp_id * 32 * 16 + lane_id * 8;
    cp_src_1 = weight + offset + index_id_1 * embedding_dim;
    cp_dst_1 = cp_dst_0 + 32 * 8;
    wmma::load_matrix_sync(
    reinterpret_cast<wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major>&>(b_frag),
    reinterpret_cast<half*>(s_masked_score) + (i - 16) * 8, 8);
    __pipeline_wait_prior(0);
    wmma::load_matrix_sync(
    reinterpret_cast<wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::col_major>&>(a_frag),
    reinterpret_cast<half*>(s_table) + warp_id * 32 * 16, 32);
    __pipeline_memcpy_async_size16_with_zero_pad(cp_dst_0, cp_src_0, each_core_offset < each_core_dim && i + (lane_id >> 2) < reduce_dim && (!has_padding_idx || org_index_id_0 != padding_idx));
    __pipeline_memcpy_async_size16_with_zero_pad(cp_dst_1, cp_src_1, each_core_offset < each_core_dim && i + (lane_id >> 2) < reduce_dim - 8 && (!has_padding_idx || org_index_id_1 != padding_idx));
    __pipeline_commit();
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::load_matrix_sync(
    reinterpret_cast<wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::row_major>&>(b_frag),
    reinterpret_cast<half*>(s_masked_score) + (i - 16) * 8, 8);
  __pipeline_wait_prior(0);
  wmma::load_matrix_sync(
  reinterpret_cast<wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::col_major>&>(a_frag),
  reinterpret_cast<half*>(s_table) + warp_id * 32 * 16, 32);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(result_buf + warp_id * 32 * 8, c_frag, 8, wmma::mem_row_major);

  #pragma unroll
  for(int i = 0; i < 8; ++i) {
    if (i < group_size && threadIdx.x < each_core_dim) {
      output[(blockIdx.x * group_size + i) * embedding_dim + blockIdx.y * each_core_dim + threadIdx.x] = __float2bfloat16(result_buf[threadIdx.x * 8 + i]);
    }
  }
}


template<bool has_padding_idx>
__global__ void lookup_backward_fused_fp32_v4_kernel(const int* index, const __nv_bfloat16* weight, const __nv_bfloat16* score, const __nv_bfloat16* output_grad,
                                                     float* weight_grad, __nv_bfloat16* score_grad, const int vocab_size, const int per_layer_vocab_size, const int shift,
                                                     const int reduce_dim, const int reduce_dim_align, const int embedding_dim, const int each_core_dim, const int group_size, const int padding_idx) {
  extern __shared__ char smem[];
  int* s_index = (int*)smem;
  int* s_group = (int*)(smem + reduce_dim_align * 4);
  int* s_org_index = (int*)(smem + reduce_dim_align * 8);
  float* s_score = (float*)(smem + reduce_dim_align * 12);
  float* s_output_grad = (float*)(smem + reduce_dim_align * 16);
  const int warp_count = blockDim.x >> 5;
  const int warp_id = threadIdx.x >> 5;
  const int warp_tid = threadIdx.x & 0x1F;
  for (int tid = threadIdx.x; tid < reduce_dim; tid += blockDim.x) {
    s_score[tid] = __bfloat162float(__ldg(&score[(blockIdx.y * gridDim.x + blockIdx.x) * reduce_dim + tid]));
    int index_id = __ldg(&index[blockIdx.x * reduce_dim + tid]);
    if (has_padding_idx) {
      s_org_index[tid] = index_id;
    }
    s_group[tid] =  index_id / per_layer_vocab_size;
    s_index[tid] = (index_id % per_layer_vocab_size + shift) % vocab_size;
  }

  int tid = threadIdx.x * 4;
  #pragma unroll
  for(int i = 0; i < 8; ++i) {
    if (i < group_size && tid < each_core_dim) {
      store_vector(&s_output_grad[i*each_core_dim+tid], load_vector(&output_grad[(blockIdx.x * group_size + i) * embedding_dim + blockIdx.y * each_core_dim + tid]));
    }
  }
  __syncthreads();
  for (int wid = warp_id; wid < reduce_dim; wid += warp_count) {
    if (has_padding_idx && s_org_index[wid] == padding_idx) {
      continue;
    }
    int64_t index_id = s_index[wid];
    int group_id = s_group[wid];
    float t_score = s_score[wid];
    for (int i = warp_tid; i < each_core_dim; i += 32) {
      float temp = t_score * s_output_grad[group_id * each_core_dim + i];
      atomicAdd(&weight_grad[index_id * embedding_dim + blockIdx.y * each_core_dim + i], temp);
    }
  }
  for (int tid = threadIdx.x; tid < reduce_dim; tid += blockDim.x) {
    float v = 0.0f;
    if (!has_padding_idx || s_org_index[tid] != padding_idx) {
      int64_t index_id = s_index[tid];
      int group_id = s_group[tid];
      float* cur_group_s_output_grad = s_output_grad + group_id * each_core_dim;
      for (int i = 0; i < each_core_dim; i += 4) {
        float4 v1 = load_vector(weight + index_id * embedding_dim + blockIdx.y * each_core_dim + i);
        float4 v2 = load_vector(cur_group_s_output_grad + i);
        v += v1.x * v2.x;
        v += v1.y * v2.y;
        v += v1.z * v2.z;
        v += v1.w * v2.w;
      }
    }
    score_grad[(blockIdx.y * gridDim.x + blockIdx.x) * reduce_dim + tid] = __float2bfloat16(v);
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


torch::Tensor EmbedLookupReduceForward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& score,
                                       int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx) {
  TORCH_INTERNAL_ASSERT(input.scalar_type() == at::ScalarType::Int);
  TORCH_INTERNAL_ASSERT(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(score.scalar_type() == at::ScalarType::BFloat16);
  std::vector<int64_t> input_shape;
  for (int i = 0; i < input.dim(); ++i) {
    input_shape.push_back(input.size(i));
  }
  std::vector<int64_t> weight_shape;
  for (int i = 0; i < weight.dim(); ++i) {
    weight_shape.push_back(weight.size(i));
  }
  std::vector<int64_t> score_shape;
  for (int i = 0; i < score.dim(); ++i) {
    score_shape.push_back(score.size(i));
  }
  const int batch_size = input_shape[0];
  const int  tucker_core_num = score_shape.size() == 2 ? 1 : score_shape[0];//score(2, batch, knn)
  const int reduce_dim = input_shape[1];
  const int reduce_dim_align = reduce_dim + 15 & -16;
  const int embedding_dim = weight_shape[1];
  int each_core_dim = embedding_dim / tucker_core_num;

  std::vector<int64_t> output_shape = input_shape;
  output_shape.back() = group_size * embedding_dim;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  auto act_options = weight.options().requires_grad(false);
  torch::Tensor output = torch::empty(output_shape, act_options);
  TORCH_INTERNAL_ASSERT(group_size<=8);
  TORCH_INTERNAL_ASSERT(each_core_dim % 8 == 0);
  const int shmem_size = sizeof(T) * reduce_dim_align + sizeof(int) * 2 * reduce_dim_align + sizeof(T) * (32 * each_core_dim) + sizeof(float4) * reduce_dim_align;
  TORCH_INTERNAL_ASSERT(sufficient(shmem_size));
  dim3 grid(batch_size, tucker_core_num, 1);
  int grid_size = batch_size;
  int block_size = each_core_dim + 31 & -32;
  if (has_padding_idx) {
    cudaFuncSetAttribute(lookup_reduce_forward_wmma_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    lookup_reduce_forward_wmma_kernel<true><<<grid, block_size, shmem_size, stream>>>(
      input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(score.data<T>()), reinterpret_cast<__nv_bfloat16*>(output.data<T>()),
      batch_size, vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, each_core_dim, group_size, padding_idx);
  } else {
    cudaFuncSetAttribute(lookup_reduce_forward_wmma_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    lookup_reduce_forward_wmma_kernel<false><<<grid, block_size, shmem_size, stream>>>(
      input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(score.data<T>()), reinterpret_cast<__nv_bfloat16*>(output.data<T>()),
      batch_size, vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, each_core_dim, group_size, padding_idx);
  }
  return output;
}

torch::Tensor EmbedLookupReduceBackward(const torch::Tensor& input, const torch::Tensor& weight,
                                        const torch::Tensor& score, const torch::Tensor& output_grad_, torch::Tensor &weight_grad,
                                        int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx) {
  TORCH_INTERNAL_ASSERT(input.scalar_type() == at::ScalarType::Int);
  TORCH_INTERNAL_ASSERT(weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(score.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(output_grad_.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(weight_grad.scalar_type() == at::ScalarType::Float);
  torch::Tensor output_grad = output_grad_.contiguous();
  std::vector<int64_t> input_shape;
  for (int i = 0; i < input.dim(); ++i) {
    input_shape.push_back(input.size(i));
  }
  std::vector<int64_t> weight_shape;
  for (int i = 0; i < weight.dim(); ++i) {
    weight_shape.push_back(weight.size(i));
  }
  std::vector<int64_t> score_shape;
  for (int i = 0; i < score.dim(); ++i) {
    score_shape.push_back(score.size(i));
  }
  const int batch_size = input_shape[0];
  const int reduce_dim = input_shape[1];
  const int  tucker_core_num = score_shape.size() == 2 ? 1 : score_shape[0];//score(2, batch, knn)
  const int reduce_dim_align = reduce_dim + 15 & -16;
  const int embedding_dim = weight_shape[1];
  //std::vector<int64_t> weight_grad_shape = weight_shape;
  std::vector<int64_t> score_grad_shape = score_shape;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  auto act_options = weight.options().requires_grad(false);
  auto score_act_options = score.options().requires_grad(false);
  torch::Tensor score_grad = torch::empty(score_grad_shape, score_act_options);

  const int shmem_size = sizeof(float) * (reduce_dim_align + group_size * embedding_dim) + sizeof(int) * 3 * reduce_dim_align;
  dim3 grid(batch_size, tucker_core_num, 1);
  int each_core_dim = embedding_dim / tucker_core_num;
  int block_size = each_core_dim + 31 & -32;
  TORCH_INTERNAL_ASSERT(each_core_dim % 4 == 0);
  if (has_padding_idx) {
    cudaFuncSetAttribute(lookup_backward_fused_fp32_v4_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    lookup_backward_fused_fp32_v4_kernel<true><<<grid, block_size, shmem_size, stream>>>(
      input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(score.data<T>()), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()),
      reinterpret_cast<float*>(weight_grad.data<float>()), reinterpret_cast<__nv_bfloat16*>(score_grad.data<T>()),
      vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, each_core_dim, group_size, padding_idx);
  } else {
    cudaFuncSetAttribute(lookup_backward_fused_fp32_v4_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    lookup_backward_fused_fp32_v4_kernel<false><<<grid, block_size, shmem_size, stream>>>(
      input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(weight.data<T>()), reinterpret_cast<const __nv_bfloat16*>(score.data<T>()), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()),
      reinterpret_cast<float*>(weight_grad.data<float>()), reinterpret_cast<__nv_bfloat16*>(score_grad.data<T>()),
      vocab_size, per_layer_vocab_size, shift, reduce_dim, reduce_dim_align, embedding_dim, each_core_dim, group_size, padding_idx);
  }
  return score_grad;
}

