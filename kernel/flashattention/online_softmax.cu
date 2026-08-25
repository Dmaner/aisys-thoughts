#include <iostream>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#define WRAP_SIZE 32

template <typename T>
__device__ T warpReduceSum(T val)
{
    for (int mask = WRAP_SIZE / 2; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__device__ T warpReduceMax(T val)
{
    for (int mask = WRAP_SIZE / 2; mask > 0; mask >>= 1)
    {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename T>
struct OnlineSoftmaxState
{
    T m;
    T dv;
};

template <typename T>
__device__ OnlineSoftmaxState<T> warpReduceSoftmaxState(OnlineSoftmaxState<T> state)
{
    for (int mask = WRAP_SIZE / 2; mask > 0; mask >>= 1)
    {
        OnlineSoftmaxState<T> other{
            __shfl_xor_sync(0xffffffff, state.m, mask),
            __shfl_xor_sync(0xffffffff, state.dv, mask),
        };
        T new_max = fmaxf(state.m, other.m);
        state.dv = state.dv * exp(state.m - new_max) + other.dv * exp(other.m - new_max);
        state.m = new_max;
    }
    return state;
}

template <typename T>
__device__ T blockReduceSum(T val)
{
    int tid = threadIdx.x;
    int wrap_id = tid / WRAP_SIZE;
    int lane_id = tid % WRAP_SIZE;
    int num_wraps = (blockDim.x + WRAP_SIZE - 1) / WRAP_SIZE;
    static __shared__ T mem[64];
    val = warpReduceSum<T>(val);
    if (lane_id == 0)
    {
        mem[wrap_id] = val;
    }
    __syncthreads();

    T sum = tid < num_wraps ? mem[tid] : (T)0;
    return warpReduceSum<T>(sum);
}

template <typename T>
__device__ T blockReduceMax(T val)
{
    int tid = threadIdx.x;
    int warp_id = tid / WRAP_SIZE;
    int lane_id = tid % WRAP_SIZE;
    int num_wraps = (blockDim.x + WRAP_SIZE - 1) / WRAP_SIZE;
    static __shared__ T mem[64];
    val = warpReduceMax<T>(val);
    if (lane_id == 0)
    {
        mem[warp_id] = val;
    }
    __syncthreads();

    T max_value = tid < num_wraps ? mem[tid] : (T)-FLT_MAX;
    return warpReduceMax<T>(max_value);
}

// d_in : [Batch, N]
// d_out: [Batch, N]
__global__ void safe_softmax(
    float *d_in,
    float *d_out,
    int batch,
    int n)
{
    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    if (batch_id >= batch)
    {
        return;
    }

    // row-wise reduce max
    float local_max = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x)
    {
        local_max = fmaxf(local_max, d_in[batch_id * n + i]);
    }
    float m = blockReduceMax<float>(local_max);

    // store max value to share memory
    __shared__ float shared_m;
    if (tid == 0)
    {
        shared_m = m;
    }
    __syncthreads();

    // get value & index
    float val = 0;
    for (int i = tid; i < n; i += blockDim.x)
    {
        int index = batch_id * n + i;
        val += exp(d_in[index] - shared_m);
    }

    // sum d_v
    float d_v = blockReduceSum<float>(val);
    __shared__ float shared_d_v;
    if (tid == 0)
    {
        shared_d_v = d_v;
    }
    __syncthreads();

    // row-wise softmax
    for (int i = tid; i < n; i += blockDim.x)
    {
        int index = batch_id * n + i;
        d_out[index] = exp(d_in[index] - shared_m) / shared_d_v;
    }
}

__global__ void online_softmax(
    float *d_in,
    float *d_out,
    int batch,
    int n)
{
    // init
    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    float m = -FLT_MAX;
    float dv = 0;

    // cross block
    for (int i = tid; i < n; i += blockDim.x)
    {
        float new_max = fmaxf(m, d_in[i + batch_id * n]);
        dv = dv * exp(m - new_max) + exp(d_in[i + batch_id * n] - new_max);
        m = new_max;
    }

    // block reduce softmax
    int warp_id = tid / WRAP_SIZE;
    int lane_id = tid % WRAP_SIZE;
    int num_warps = (blockDim.x + WRAP_SIZE - 1) / WRAP_SIZE;
    static __shared__ float shared_m[64];
    static __shared__ float shared_dv[64];

    OnlineSoftmaxState<float> state{m, dv};
    state = warpReduceSoftmaxState<float>(state);

    // store to shared mem
    if (lane_id == 0)
    {
        shared_m[warp_id] = state.m;
        shared_dv[warp_id] = state.dv;
    }
    __syncthreads();

    // final warp reduce
    float final_m = tid < num_warps ? shared_m[tid] : -FLT_MAX;
    float final_dv = tid < num_warps ? shared_dv[tid] : 0;
    OnlineSoftmaxState<float> final_state{final_m, final_dv};

    final_state = warpReduceSoftmaxState(final_state);

    if (tid == 0)
    {
        shared_m[0] = final_state.m;
        shared_dv[0] = final_state.dv;
    }
    __syncthreads();

    // row-wise calculate softmax
    for (int i = tid; i < n; i += blockDim.x)
    {
        d_out[i + batch_id * n] = exp(d_in[i + batch_id * n] - shared_m[0]) / shared_dv[0];
    }
}

// PyTorch launcher: Tensor -> raw pointer -> CUDA kernel -> Tensor.
at::Tensor run_online_softmax(const at::Tensor &input)
{
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [B, N]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int batch = static_cast<int>(input.size(0));
    int n = static_cast<int>(input.size(1));
    auto output = at::empty_like(input);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    online_softmax<<<batch, 256, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        n);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

TORCH_LIBRARY(online_softmax_ops, m)
{
    m.def("forward(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(online_softmax_ops, CUDA, m)
{
    m.impl("forward", &run_online_softmax);
}
