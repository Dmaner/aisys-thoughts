#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WRAPSIZE 32

template <typename T>
__device__ __forceinline__ T warpReduce(T val)
{
    for (int mask = WRAPSIZE >> 1; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// dst[i] = sum_j(vec * mat[i])
void gemvCPU(half *vec, half *mat, float *dst, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += __half2float(vec[j]) * __half2float(mat[i * M + j]);
        }
        dst[i] = sum;
    }
}

bool check_results(half *d_dst, float *dst, int N)
{
    for (int i = 0; i < N; i++)
    {
        if ((fabsf(__half2float(d_dst[i]) - dst[i]) / fabsf(dst[i])) > 1e-2)
        {
            return false;
        }
    }
    return true;
}

__global__ void gemv_v0(half *vec, half *mat, half *dst, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_sum = 0.0;
    if (row < M)
    {
        for (int j = 0; j < N; j++)
        {
            thread_sum = fmaf(
                __half2float(vec[j]),
                __half2float(mat[row * N + j]),
                thread_sum);
        }
        dst[row] = thread_sum;
    }
}

union __align__(16) Half8
{
    float4 value;
    half half_value[8];
};

// using float4
__global__ void gemv_v1(half *vec, half *mat, half *dst, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_sum = 0.0;

    if (row < M)
    {
        const float4 *vec4 = reinterpret_cast<const float4 *>(vec);
        const float4 *mat4 = reinterpret_cast<const float4 *>(mat + row * N);
        for (int j = 0; j < N / 8; j++)
        {
            Half8 v, m;
            v.value = vec4[j];
            m.value = mat4[j];
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                float vf = __half2float(v.half_value[k]);
                float mf = __half2float(m.half_value[k]);
                thread_sum = fmaf(vf, mf, thread_sum);
            }
        }
        dst[row] = __float2half(thread_sum);
    }
}

// block size 256, WARPS_PER_BLOCK = block size / warp size = 8, grid_size = N / WARPS_PER_BLOCK = 512
__global__ void gemv_v2(
    const half *__restrict__ vec,
    const half *__restrict__ mat,
    half *__restrict__ dst,
    int M,
    int N)
{
    int lane_id = threadIdx.x % WRAPSIZE;
    int warp_id = threadIdx.x / WRAPSIZE;
    int warps_per_block = blockDim.x / WRAPSIZE;

    int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= M)
        return;

    float thread_sum = 0.0f;
    const float4 *vec4 = reinterpret_cast<const float4 *>(vec);
    const float4 *mat4 = reinterpret_cast<const float4 *>(mat + row * N);
    int vec_count = N / 8;

    for (int j = lane_id; j < vec_count; j += WRAPSIZE)
    {
        Half8 v;
        Half8 m;

        v.value = vec4[j];
        m.value = mat4[j];

#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float vf = __half2float(v.half_value[k]);
            float mf = __half2float(m.half_value[k]);
            thread_sum = fmaf(vf, mf, thread_sum);
        }
    }
    thread_sum = warpReduce<float>(thread_sum);
    if (lane_id == 0)
    {
        dst[row] = __float2half(thread_sum);
    }
}

int main()
{
    float milliseconds = 0;
    int N = 4096;
    half *vec;
    half *mat;
    half *dst;
    half *d_vec;
    half *d_mat;
    half *d_dst;
    // malloc
    vec = (half *)malloc(N * sizeof(half));
    cudaMalloc((void **)&d_vec, N * sizeof(half));
    mat = (half *)malloc(N * N * sizeof(half));
    cudaMalloc((void **)&d_mat, N * N * sizeof(half));
    dst = (half *)malloc(N * sizeof(half));
    cudaMalloc((void **)&d_dst, N * sizeof(half));
    for (int i = 0; i < N; i++)
    {
        vec[i] = (half)1;
    }
    for (int i = 0; i < N * N; i++)
    {
        mat[i] = (half)1;
    }
    // memory copy
    cudaMemcpy(d_vec, vec, sizeof(half) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat, sizeof(half) * N * N, cudaMemcpyHostToDevice);

    // warm up 10
    for (int i = 0; i < 10; ++i)
    {
        // gemv_v0<<<4, 1024>>>(d_vec, d_mat, d_dst, N, N);
        // gemv_v1<<<4, 1024>>>(d_vec, d_mat, d_dst, N, N);
        gemv_v2<<<512, 256>>>(d_vec, d_mat, d_dst, N, N);
    }

    // launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i)
    {
        // gemv_v0<<<4, 1024>>>(d_vec, d_mat, d_dst, N, N);
        // gemv_v1<<<4, 1024>>>(d_vec, d_mat, d_dst, N, N);
        gemv_v2<<<512, 256>>>(d_vec, d_mat, d_dst, N, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(dst, d_dst, sizeof(half) * N, cudaMemcpyDeviceToHost);

    // check result
    float *fp32_dst = (float *)malloc(sizeof(float) * N);

    gemvCPU(vec, mat, fp32_dst, N, N);
    if (check_results(dst, fp32_dst, N))
    {
        printf("the answer is correct!\n");
    }
    else
    {
        printf("the answer is wrong!\n");
    }
    printf("gemv latency = %f ms\n", milliseconds / 100);

    // free
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
    free(vec);
    free(mat);
    free(dst);
    free(fp32_dst);
}