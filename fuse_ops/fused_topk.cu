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

union b16 {
  uint16_t u16;
  struct {
    uint8_t lo;
    uint8_t hi;
  } u8;
};

__device__ __forceinline__ unsigned get_warp_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ unsigned get_lane_id() {
  unsigned ret;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ void __pipeline_memcpy_async_size4_with_zero_pad(void* smem, const void* gmem, int pred) {
  int s = __cvta_generic_to_shared(smem);
  asm volatile(
    "{.reg .pred p;\n\t"
    "setp.eq.u32 p, %0, 0;\n\t"
    "cp.async.ca.shared.global [%1], [%2], 4, p;}\n\t" : : "r"(pred), "r"(s), "l"(gmem));
}

__device__ __forceinline__ void __pipeline_memcpy_async_size8_with_zero_pad(void* smem, const void* gmem, int pred) {
  int s = __cvta_generic_to_shared(smem);
  asm volatile(
    "{.reg .pred p;\n\t"
    "setp.eq.u32 p, %0, 0;\n\t"
    "cp.async.ca.shared.global [%1], [%2], 8, p;}\n\t" : : "r"(pred), "r"(s), "l"(gmem));
}

__device__ __forceinline__ void __pipeline_memcpy_async_size16_with_zero_pad(void* smem, const void* gmem, int pred) {
  int s = __cvta_generic_to_shared(smem);
  asm volatile(
    "{.reg .pred p;\n\t"
    "setp.eq.u32 p, %0, 0;\n\t"
    "cp.async.cg.shared.global [%1], [%2], 16, p;}\n\t" : : "r"(pred), "r"(s), "l"(gmem));
}

__device__ __forceinline__ void warp8_cumsum(int32_t tid, int32_t* table, int32_t* value_index_ptr, int32_t* value_count_ptr, int32_t sum_offset) {
  __syncthreads();
  int32_t count;
  int32_t sum;
  if (tid < 256) {
    sum = count = table[tid];
    *value_index_ptr = 256;
    #pragma unroll
    for (int i = 1; i < 32; i *= 2) {
      int32_t pre = __shfl_up_sync(~0, sum, i);
      if ((tid & 31) >= i) {
        sum += pre;
      }
    }
    table[tid] = sum;
  }
  __syncthreads();
  if (tid < 256) {
    #pragma unroll
    for (int i = 0; i < 7; i++) {
      if (i < (tid >> 5)) {
        sum += table[i * 32 + 31];
      }
    }
  }
  __syncthreads();
  if (tid < 256) {
    sum += sum_offset;
    if (sum >= 0 && sum < count) {
      *value_index_ptr = tid;
      *value_count_ptr = count;
    }
    table[tid] = sum - count;
  }
  __syncthreads();
}

template <int32_t num_per_thread, bool wait_pipeline>
__device__ void fast_radix_topk_16bit(int32_t tid, int32_t nt, const nv_bfloat16* in_buf, int32_t* table1, int32_t* table2, int32_t* useless, int32_t* high, int32_t* high_count, int32_t* low, int32_t* low_count, int32_t* index_buf, int32_t n, int32_t k) {
  if (tid < 256) {
    table1[tid] = 0;
    table2[tid] = 0;
  }
  int32_t lane = get_lane_id();
  if (wait_pipeline) {
    __pipeline_wait_prior(0);
  }
  __syncthreads();
  uint32_t value_hi[num_per_thread];
  uint32_t value_lo[num_per_thread];
  #pragma unroll
  for (int32_t i = 0; i < num_per_thread; i++) {
    uint32_t value = *((const uint16_t*)&in_buf[i * nt + tid]);
    if (value & 0x8000) {
      value ^= 0xffff;
    } else {
      value |= 0x8000;
    }
    value_hi[i] = value >> 8;
    value_lo[i] = value & 0xff;
    atomicAdd(&table1[value_hi[i]], 1);
  }
  warp8_cumsum(tid, table1, high, high_count, k - n);
  int32_t high_value = *high;
  #pragma unroll
  for (int32_t i = 0; i < num_per_thread; i++) {
    bool maybe_topk = (value_hi[i]) == high_value;
    bool is_topk = (value_hi[i]) > high_value;
    int32_t index = atomicAdd(maybe_topk ? &table2[value_lo[i]] : (is_topk ? &table1[value_hi[i]] : &useless[lane]), 1);
    if (is_topk) index_buf[index] = i * nt + tid;
  }
  __syncthreads();
  warp8_cumsum(tid, table2, low, low_count, table1[high_value]);
  int32_t low_value = *low;
  #pragma unroll
  for (int32_t i = 0; i < num_per_thread; i++) {
    uint32_t hi = value_hi[i];
    uint32_t lo = value_lo[i];
    int32_t idx = i * nt + tid;

    bool is_topk = hi == high_value && lo >= low_value;
    // int32_t index = atomicAdd(is_topk ? &table2[value_lo[i]] : &useless[lane], 1);
    int p0 = is_topk;
    int s1 = __cvta_generic_to_shared(is_topk ? &table2[value_lo[i]] : &useless[lane]);
    int s2 = __cvta_generic_to_shared(index_buf);
    asm volatile(
        "{.reg .pred p0, p1;\n\t"
        ".reg .s32 index, sts_addr;\n\t"
        "setp.ne.s32 p0, %0, 0;\n\t"
        "atom.shared.add.s32 index, [%1], 1;\n\t"
        "setp.ge.and.s32 p1, index, 0, p0;\n\t"
        "mad.lo.s32 sts_addr, index, 4, %2;\n\t"
        "@p1 st.shared.s32 [sts_addr], %3;}\n\t" : : "r"(p0), "r"(s1), "r"(s2), "r"(idx));
  }
  __syncthreads();
}

template<typename T>
__device__ void einsum_step1_rank2_func(T* output, const T* scores, float2 c, const int n_keys) {
  for (int k = threadIdx.x; k < n_keys; k += blockDim.x) {
    float s0, s1, c0, c1;
    if constexpr (sizeof(T) == 2) {
      union Vec2
      {
        T arr[2];
        float vec;
      } s = ((Vec2*)scores)[k];
      s0 = ConvertTo<float>(s.arr[0]);
      s1 = ConvertTo<float>(s.arr[1]);
    }
    c0 = c.x;
    c1 = c.y;
    float sum = s0 * c0 + s1 * c1;
    output[k] = __float2bfloat16(sum);
  }
}

template<typename T, bool output_global>
__device__ void einsum_step2_rank2_func(T* output, T* output_sum, const T* scores1, const T* scores2, const float* multi_tucker_core, const int k1, const int tucker_core_num) {
  const int tucker_rank = 2;
  int batch_size = gridDim.y;
  int head_num = gridDim.x;

  for (int y = threadIdx.y; y < k1; y += blockDim.y) {
    for (int x = threadIdx.x; x < k1; x += blockDim.x) {
      float outer_sum = 0.0f;
      for (int tucker_core = 0; tucker_core < tucker_core_num; tucker_core++) {
        const float* tucker_core_ptr = multi_tucker_core + (tucker_core * head_num) * (tucker_rank * tucker_rank);
        T* output_ptr;
        if (output_global) {
          output_ptr = output + int64_t(tucker_core * batch_size * head_num) * (k1 * k1);
        } else {
          output_ptr = output + tucker_core * (k1 * k1);
        }

        float s1_0, s1_1, s2_0, s2_1, c00, c01, c10, c11;
        float4 c = *(float4*)tucker_core_ptr;
        // float s1_0 = scores1[y * tucker_rank + 0];
        // float s1_1 = scores1[y * tucker_rank + 1];
        // float s2_0 = scores2[x * tucker_rank + 0];
        // float s2_1 = scores2[x * tucker_rank + 1];
        if constexpr (sizeof(T) == 2) {
          union Vec2
          {
            T arr[2];
            float vec;
          } s1, s2;
          s1 = ((Vec2*)scores1)[y];
          s2 = ((Vec2*)scores2)[x];
          s1_0 = ConvertTo<float>(s1.arr[0]);
          s1_1 = ConvertTo<float>(s1.arr[1]);
          s2_0 = ConvertTo<float>(s2.arr[0]);
          s2_1 = ConvertTo<float>(s2.arr[1]);
        }
        c00 = c.x;
        c01 = c.y;
        c10 = c.z;
        c11 = c.w;
        float inner_sum = s1_0 * s2_0 * c00;
        inner_sum += s1_0 * s2_1 * c01;
        inner_sum += s1_1 * s2_0 * c10;
        inner_sum += s1_1 * s2_1 * c11;
        output_ptr[y * k1 + x] = __float2bfloat16(inner_sum);
        outer_sum += inner_sum;
      }
      output_sum[y * k1 + x] = __float2bfloat16(outer_sum);
    }
  }
}

template<int32_t num_per_thread, bool has_balance_loss>
__global__ __launch_bounds__(1024, 1)
void fused_einsum_topk_step1_rank2_kernel(nv_bfloat16* scores_out, nv_bfloat16* value_out, int32_t* index_out, int32_t* ngram, const nv_bfloat16* scores, const float* tucker_core_uv, const int n_keys, const int k1) {
  const int tucker_rank = 2;
  int key_set = blockIdx.z;
  // int key_set_num = gridDim.z;
  int batch = blockIdx.y;
  int batch_size = gridDim.y;
  int head = blockIdx.x;
  int head_num = gridDim.x;

  extern __shared__ char smem[];
  int32_t* table1 = (int32_t*)smem;
  int32_t* table2 = (int32_t*)(smem + 1024);
  int32_t* useless = (int32_t*)(smem + 2048);
  int32_t* high = (int32_t*)(smem + 2176);
  int32_t* high_count = (int32_t*)(smem + 2180);
  int32_t* low = (int32_t*)(smem + 2184);
  int32_t* low_count = (int32_t*)(smem + 2188);
  int32_t* index_buf = (int32_t*)(smem + 2560);
  // float* tucker_core_uv_buf = (float*)(smem + 2560);
  nv_bfloat16* scores_in_buf = (nv_bfloat16*)(smem + 4096);
  nv_bfloat16* scores_out_buf = (nv_bfloat16*)(smem + 4096 + n_keys * 4);

  scores += ((key_set * batch_size + batch) * head_num + head) * int64_t(n_keys * tucker_rank);
  tucker_core_uv += (key_set * head_num + head) * tucker_rank;
  value_out += ((key_set * batch_size + batch) * head_num + head) * (int64_t)k1 * tucker_rank;
  index_out += ((key_set * batch_size + batch) * head_num + head) * (int64_t)k1;
  if constexpr (has_balance_loss) {
    scores_out += ((key_set * batch_size + batch) * head_num + head) * (int64_t)n_keys;
    ngram += key_set * n_keys;
  }

  float2 c = *(float2*)tucker_core_uv;
  #pragma unroll
  for (int32_t i = 0; i < num_per_thread; i++) {
    if (i * blockDim.x + threadIdx.x < n_keys) {
      __pipeline_memcpy_async(&scores_in_buf[(i * blockDim.x + threadIdx.x) * 2], &scores[(i * blockDim.x + threadIdx.x) * 2], 4);
    } else {
      *(uint16_t*)&scores_out_buf[i * blockDim.x + threadIdx.x] = 0xffff;
    }
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  einsum_step1_rank2_func(scores_out_buf, scores_in_buf, c, n_keys);

  fast_radix_topk_16bit<num_per_thread, false>(threadIdx.x, blockDim.x, scores_out_buf, table1, table2, useless, high, high_count, low, low_count, index_buf, num_per_thread * blockDim.x, k1);
  for (int32_t i = threadIdx.x; i < k1; i += blockDim.x) {
    int32_t idx = index_buf[i];
    if constexpr (has_balance_loss) {
      atomicAdd(&ngram[idx], 1);
    }
    index_out[i] = idx;
    *(int*)&value_out[i*2] = *(int*)&scores_in_buf[idx*2];
  }
  if constexpr (has_balance_loss) {
    for (int32_t i = threadIdx.x; i < n_keys; i += blockDim.x) {
      scores_out[i] = scores_out_buf[i];
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> fused_einsum_topk_step1(const torch::Tensor& scores, const torch::Tensor& tucker_core_uv, int64_t k1) {
  auto ket_set_num = scores.size(0);
  auto batch_size = scores.size(1);
  auto head_num = scores.size(2);
  auto n_keys = scores.size(3);
  auto tucker_rank = scores.size(4);

  TORCH_INTERNAL_ASSERT(tucker_core_uv.size(0) == ket_set_num);
  TORCH_INTERNAL_ASSERT(tucker_core_uv.size(1) == head_num);
  TORCH_INTERNAL_ASSERT(tucker_core_uv.size(2) == tucker_rank);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  std::vector<int64_t> value_out_shape = { ket_set_num, batch_size, head_num, k1, tucker_rank };
  std::vector<int64_t> index_out_shape = { ket_set_num, batch_size, head_num, k1 };
  auto act_options = scores.options().requires_grad(false);
  torch::Tensor value_out = torch::empty(value_out_shape, act_options);
  torch::Tensor index_out = torch::empty(index_out_shape, act_options.dtype(torch::kInt32));

  const int nt = std::max(256, int((n_keys + 127) / 128) * 8);
  const int shmem_size = 4096 + sizeof(T) * nt * 16 * 3;
  cudaFuncSetAttribute(fused_einsum_topk_step1_rank2_kernel<16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
  fused_einsum_topk_step1_rank2_kernel<16, false><<<dim3(head_num, batch_size, ket_set_num), nt, shmem_size, stream>>>(
    nullptr, (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), nullptr, (nv_bfloat16*)scores.data_ptr(), (float*)tucker_core_uv.data_ptr(), n_keys, k1);
  return std::make_tuple(value_out, index_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_einsum_topk_step1_balance_loss(const torch::Tensor& scores, const torch::Tensor& tucker_core_uv, int64_t k1) {
  auto ket_set_num = scores.size(0);
  auto batch_size = scores.size(1);
  auto head_num = scores.size(2);
  auto n_keys = scores.size(3);
  auto tucker_rank = scores.size(4);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  std::vector<int64_t> scores_out_shape = { ket_set_num, batch_size, head_num, n_keys };
  std::vector<int64_t> value_out_shape = { ket_set_num, batch_size, head_num, k1, tucker_rank };
  std::vector<int64_t> index_out_shape = { ket_set_num, batch_size, head_num, k1 };
  std::vector<int64_t> ngram_shape = { ket_set_num, n_keys };
  auto act_options = scores.options().requires_grad(false);
  torch::Tensor scores_out = torch::empty(scores_out_shape, act_options);
  torch::Tensor value_out = torch::empty(value_out_shape, act_options);
  torch::Tensor index_out = torch::empty(index_out_shape, act_options.dtype(torch::kInt32));
  torch::Tensor ngram = torch::empty(ngram_shape, act_options.dtype(torch::kInt32));

  cudaMemsetAsync(ngram.data_ptr(), 0, ket_set_num * n_keys * sizeof(int32_t), stream);
  const int nt = std::max(256, int((n_keys + 127) / 128) * 8);
  const int shmem_size = 4096 + sizeof(T) * nt * 16 * 3;
  cudaFuncSetAttribute(fused_einsum_topk_step1_rank2_kernel<16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
  fused_einsum_topk_step1_rank2_kernel<16, true><<<dim3(head_num, batch_size, ket_set_num), nt, shmem_size, stream>>>(
    (nv_bfloat16*)scores_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (int32_t*)ngram.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)tucker_core_uv.data_ptr(), n_keys, k1);
  return std::make_tuple(scores_out, value_out, index_out, ngram);
}

template<int k1, int tucker_core_num>
__global__ __launch_bounds__(1024, 1)
void fused_einsum_topk_step2_rank2_kernel(nv_bfloat16* value_sum_out, nv_bfloat16* value_out, int32_t* index_out, const nv_bfloat16* scores, const float* multi_tucker_core, const int k2) {
  const int tucker_rank = 2;
  const int n = k1 * k1;
  int batch = blockIdx.y;
  int batch_size = gridDim.y;
  int head = blockIdx.x;
  int head_num = gridDim.x;

  extern __shared__ char smem[];
  int32_t* table1 = (int32_t*)smem;
  int32_t* table2 = (int32_t*)(smem + 1024);
  int32_t* useless = (int32_t*)(smem + 2048);
  int32_t* high = (int32_t*)(smem + 2176);
  int32_t* high_count = (int32_t*)(smem + 2180);
  int32_t* low = (int32_t*)(smem + 2184);
  int32_t* low_count = (int32_t*)(smem + 2188);
  int32_t* index_buf = (int32_t*)(smem + 2560);
  float* multi_tucker_core_buf = (float*)(smem + 2560);
  nv_bfloat16* scores1_buf = (nv_bfloat16*)(smem + 3072);
  nv_bfloat16* scores2_buf = (nv_bfloat16*)(smem + 3584);
  nv_bfloat16* all_scores_buf = (nv_bfloat16*)(smem + 4096);
  nv_bfloat16* score_list_buf = (nv_bfloat16*)(smem + 4096 + n * 2);
  
  value_sum_out += batch * (int64_t)k2;
  value_out += batch * (int64_t)k2;
  index_out += batch * (int64_t)k2;
  multi_tucker_core += head * (tucker_rank * tucker_rank);
  const nv_bfloat16* scores1 = scores + (batch * head_num + head) * int64_t(k1 * tucker_rank);
  const nv_bfloat16* scores2 = scores1 + (batch_size * head_num) * (k1 * tucker_rank);

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < head_num * tucker_core_num; i += 1024) {
    __pipeline_memcpy_async_size16_with_zero_pad(&multi_tucker_core_buf[i*tucker_rank*tucker_rank], &multi_tucker_core[i*tucker_rank*tucker_rank], i < head_num * tucker_core_num);
  }
  for (int i = tid; i < k1; i += 1024) {
    __pipeline_memcpy_async_size4_with_zero_pad(&scores1_buf[i*tucker_rank], &scores1[i*tucker_rank], i < k1);
    __pipeline_memcpy_async_size4_with_zero_pad(&scores2_buf[i*tucker_rank], &scores2[i*tucker_rank], i < k1);  
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  einsum_step2_rank2_func<nv_bfloat16, false>(score_list_buf, all_scores_buf, scores1_buf, scores2_buf, multi_tucker_core_buf, k1, tucker_core_num);

  fast_radix_topk_16bit<n / 1024, false>(tid, 1024, all_scores_buf, table1, table2, useless, high, high_count, low, low_count, index_buf, n, k2);
  for (int32_t i = tid; i < k2; i += 1024) {
    int32_t idx = index_buf[i];
    index_out[i] = idx;
    value_sum_out[i] = all_scores_buf[idx];
    for (int tucker_core = 0; tucker_core < tucker_core_num; tucker_core++) {
      nv_bfloat16* output_ptr = value_out + int64_t(tucker_core * batch_size * head_num) * k2;
      output_ptr[i] = score_list_buf[idx + tucker_core * n];
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_einsum_topk_step2(const torch::Tensor& scores, const torch::Tensor& multi_tucker_core, int64_t k2) {
  int tucker_core_num = multi_tucker_core.size(0);
  int batch_size = scores.size(1);
  int head_num = scores.size(2);
  int k1 = scores.size(3);
  int tucker_rank = scores.size(4);

  TORCH_INTERNAL_ASSERT(tucker_rank == 2);
  TORCH_INTERNAL_ASSERT(k1 == 32 || k1 == 64 || k1 == 128);
  TORCH_INTERNAL_ASSERT(tucker_core_num == 1 || tucker_core_num == 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  using T = at::BFloat16;
  std::vector<int64_t> value_sum_out_shape = { batch_size, head_num, k2 };
  std::vector<int64_t> value_out_shape = { tucker_core_num, batch_size, head_num, k2 };
  std::vector<int64_t> index_out_shape = { batch_size, head_num, k2 };
  auto act_options = scores.options().requires_grad(false);
  torch::Tensor value_sum_out = torch::empty(value_sum_out_shape, act_options);
  torch::Tensor value_out = torch::empty(value_out_shape, act_options);
  torch::Tensor index_out = torch::empty(index_out_shape, act_options.dtype(torch::kInt32));

  const int shmem_size = 4096 + sizeof(nv_bfloat16) * (tucker_core_num + 1) * k1 * k1;
  if (k1 == 32) {
    if (tucker_core_num == 1) {
      cudaFuncSetAttribute(fused_einsum_topk_step2_rank2_kernel<32, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
      fused_einsum_topk_step2_rank2_kernel<32, 1><<<dim3(head_num, batch_size, 1), dim3(32, 32, 1), shmem_size, stream>>>(
        (nv_bfloat16*)value_sum_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)multi_tucker_core.data_ptr(), k2);
    }
    if (tucker_core_num == 2) {
      cudaFuncSetAttribute(fused_einsum_topk_step2_rank2_kernel<32, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
      fused_einsum_topk_step2_rank2_kernel<32, 2><<<dim3(head_num, batch_size, 1), dim3(32, 32, 1), shmem_size, stream>>>(
        (nv_bfloat16*)value_sum_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)multi_tucker_core.data_ptr(), k2);
    }
  } else if (k1 == 64) {
    if (tucker_core_num == 1) {
      cudaFuncSetAttribute(fused_einsum_topk_step2_rank2_kernel<64, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
      fused_einsum_topk_step2_rank2_kernel<64, 1><<<dim3(head_num, batch_size, 1), dim3(32, 32, 1), shmem_size, stream>>>(
        (nv_bfloat16*)value_sum_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)multi_tucker_core.data_ptr(), k2);
    }
    if (tucker_core_num == 2) {
      cudaFuncSetAttribute(fused_einsum_topk_step2_rank2_kernel<64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
      fused_einsum_topk_step2_rank2_kernel<64, 2><<<dim3(head_num, batch_size, 1), dim3(32, 32, 1), shmem_size, stream>>>(
        (nv_bfloat16*)value_sum_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)multi_tucker_core.data_ptr(), k2);
    }
  } else if (k1 == 128) {   
    if (tucker_core_num == 1) {
      cudaFuncSetAttribute(fused_einsum_topk_step2_rank2_kernel<128, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
      fused_einsum_topk_step2_rank2_kernel<128, 1><<<dim3(head_num, batch_size, 1), dim3(32, 32, 1), shmem_size, stream>>>(
        (nv_bfloat16*)value_sum_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)multi_tucker_core.data_ptr(), k2);
    }
    if (tucker_core_num == 2) {
      cudaFuncSetAttribute(fused_einsum_topk_step2_rank2_kernel<128, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);  
      fused_einsum_topk_step2_rank2_kernel<128, 2><<<dim3(head_num, batch_size, 1), dim3(32, 32, 1), shmem_size, stream>>>(
        (nv_bfloat16*)value_sum_out.data_ptr(), (nv_bfloat16*)value_out.data_ptr(), (int32_t*)index_out.data_ptr(), (nv_bfloat16*)scores.data_ptr(), (float*)multi_tucker_core.data_ptr(), k2);
    }
  }
  return std::make_tuple(value_sum_out, value_out, index_out);
}

#endif
