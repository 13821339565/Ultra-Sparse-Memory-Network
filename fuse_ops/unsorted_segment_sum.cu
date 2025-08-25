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
__inline__ __device__ T AtomicAdd(T* address, T val) {
  return atomicAdd(address, val);
}

template <>
__inline__ __device__ float2 AtomicAdd<float2>(float2* address, float2 val) {
#if (__CUDA_ARCH__ >= 900)
  return atomicAdd(address, val);
#else
  float2 res;
  res.x = atomicAdd(&(address->x), val.x);
  res.y = atomicAdd(&(address->y), val.y);
  return res;
#endif
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

template <typename T, typename WeightT, typename IndexType>
__global__ void unsorted_segment_sum_kernel(const int* index, const T* output_grad,
                                       WeightT* weight_grad, const IndexType out_size_half, 
                                       const int inner_dim, const int out_cnt_per_batch, const int weight_cnt_per_batch, const int out_row_dim) {
    using WeightT2 = typename native_pack<WeightT>::T2;
    using OutT2 = typename native_pack<T>::T2;
    const IndexType inner_dim_half = inner_dim / 2;
    for(IndexType out_id = blockIdx.x * blockDim.x + threadIdx.x; out_id < out_size_half; out_id += blockDim.x * gridDim.x) {
        IndexType batch_id = out_id / out_cnt_per_batch;
        IndexType out_id_in_batch = out_id - batch_id * out_cnt_per_batch;
        IndexType row_id = out_id_in_batch / inner_dim_half;
        IndexType col_id = out_id_in_batch - row_id * inner_dim_half;
        const int index_id = index[batch_id * out_row_dim + row_id];
        const int64_t offset = batch_id * weight_cnt_per_batch + static_cast<int64_t>(index_id) * inner_dim_half + col_id;
        const OutT2 output_grad_val = (reinterpret_cast<const OutT2*>(output_grad))[out_id];
        WeightT2 tmp_val;
        tmp_val.x = ConvertTo<WeightT>(output_grad_val.x);
        tmp_val.y = ConvertTo<WeightT>(output_grad_val.y);
        AtomicAdd<WeightT2>(reinterpret_cast<WeightT2*>(weight_grad) + offset, tmp_val);
    }
}

//input:(bs, out_dim) output_grad:(bs, out_dim, embedding_dim) weight_grad:(bs, gather_dim, embedding_dim)
void UnsortedSegmentSum(const torch::Tensor& input, const torch::Tensor& output_grad_, torch::Tensor &weight_grad, const int padding_idx) {
  TORCH_INTERNAL_ASSERT(input.scalar_type() == at::ScalarType::Int);
  TORCH_INTERNAL_ASSERT(output_grad_.scalar_type() == at::ScalarType::BFloat16);
  TORCH_INTERNAL_ASSERT(weight_grad.scalar_type() == at::ScalarType::BFloat16 || weight_grad.scalar_type() == at::ScalarType::Float);
  torch::Tensor output_grad = output_grad_.contiguous();
  std::vector<int64_t> input_shape;
  for (int i = 0; i < input.dim(); ++i) {
    input_shape.push_back(input.size(i));
  }
  std::vector<int64_t> weight_shape;
  for (int i = 0; i < weight_grad.dim(); ++i) {
    weight_shape.push_back(weight_grad.size(i));
  }
  const int batch_size = input_shape[0];
  const int embedding_dim = weight_shape[2];
  const int64_t out_size = output_grad.numel();
  TORCH_INTERNAL_ASSERT(embedding_dim % 2 == 0);

  auto weight_uintptr = reinterpret_cast<std::uintptr_t>(weight_grad.data_ptr());
  auto out_uintptr = reinterpret_cast<std::uintptr_t>(output_grad.data_ptr());
  TORCH_INTERNAL_ASSERT(weight_uintptr % 2 == 0);
  TORCH_INTERNAL_ASSERT(out_uintptr % 2 == 0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  const int block_size = 512;
  const int grid_size = 10240;
  const int64_t out_size_half = out_size / 2;
  const int64_t out_cnt_per_batch = out_size_half / batch_size;
  const int64_t weight_cnt_per_batch = weight_grad.numel() / 2 / batch_size;
  const int64_t out_row_dim = output_grad.size(1);
  if(weight_grad.scalar_type() == at::ScalarType::BFloat16) {
    if (out_size_half < std::numeric_limits<int32_t>::max() / 2) {
      unsorted_segment_sum_kernel<__nv_bfloat16, __nv_bfloat16, int32_t><<<grid_size, block_size, 0, stream>>>(
          input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()), reinterpret_cast<__nv_bfloat16*>(weight_grad.data<T>()),
          static_cast<int32_t>(out_size_half), embedding_dim, out_cnt_per_batch, weight_cnt_per_batch, out_row_dim);
    } else {
      unsorted_segment_sum_kernel<__nv_bfloat16, __nv_bfloat16, int64_t><<<grid_size, block_size, 0, stream>>>(
          input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()), reinterpret_cast<__nv_bfloat16*>(weight_grad.data<T>()),
          out_size_half, embedding_dim, out_cnt_per_batch, weight_cnt_per_batch, out_row_dim);
    }
  } else {
    if (out_size_half < std::numeric_limits<int32_t>::max() / 2) {
      unsorted_segment_sum_kernel<__nv_bfloat16, float, int32_t><<<grid_size, block_size, 0, stream>>>(
          input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()), reinterpret_cast<float*>(weight_grad.data<float>()), 
          static_cast<int32_t>(out_size_half), embedding_dim, out_cnt_per_batch, weight_cnt_per_batch, out_row_dim);
    } else {
      unsorted_segment_sum_kernel<__nv_bfloat16, float, int64_t><<<grid_size, block_size, 0, stream>>>(
          input.data<int>(), reinterpret_cast<const __nv_bfloat16*>(output_grad.data<T>()), reinterpret_cast<float*>(weight_grad.data<float>()), 
          out_size_half, embedding_dim, out_cnt_per_batch, weight_cnt_per_batch, out_row_dim);
    }
  }
}

#endif