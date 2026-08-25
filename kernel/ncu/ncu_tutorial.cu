// nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 \
//   ncu_tutorial.cu -o ncu_tutorial

#include <cuda_runtime.h>

#include <cstdio>

constexpr int my_L = 1024;
constexpr int my_M = 1024;
constexpr int my_N = 32;

//
template <typename T>
__global__ void gpu_version1(const T *__restrict__ input,
                             T *__restrict__ output,
                             const T *__restrict__ matrix, const int L,
                             const int M, const int N)
{
    __shared__ T smem[my_L];
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (int k = 0; k < N; ++k)
    {
        T v1 = T(0);
        for (int i = 0; i < M; ++i)
        {
            v1 += input[static_cast<size_t>(k) * M * L + idx * M + i];
        }
        v1 /= T(M);

        for (int row = 0; row < L; ++row)
        {
            __syncthreads();
            smem[threadIdx.x] =
                v1 * matrix[row * L + idx];

            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
            {
                __syncthreads();
                if (threadIdx.x < stride)
                {
                    smem[threadIdx.x] += smem[threadIdx.x + stride];
                }
            }

            if (threadIdx.x == 0)
            {
                output[row * N + k] = smem[0];
            }
        }
    }
}

// block_size(L), grid_size(N)
template <typename T>
__global__ void gpu_version2(const T *__restrict__ input,
                             T *__restrict__ output,
                             const T *__restrict__ matrix, const int L,
                             const int M, const int N)
{
    // parallelize threadIdx.x over vector length, and blockIdx.x across k (N)
    __shared__ T smem[my_L];
    int idx = threadIdx.x;
    int k = blockIdx.x;
    T v1 = 0;
    // perform vector averaging
    for (int i = 0; i < M; i++)
    {
        v1 += input[k * M * L + idx * M + i];
    }
    v1 /= M;
    for (int i = 0; i < L; i++)
    { // perform matrix-vector multiply
        __syncthreads();
        smem[threadIdx.x] = v1 * matrix[i * L + idx];
        for (int s = L >> 1; s > 0; s >>= 1)
        {
            __syncthreads();
            if (threadIdx.x < s)
            {
                smem[threadIdx.x] += smem[threadIdx.x + s];
            }
        }
        if (!threadIdx.x)
        {
            output[k + i * N] = smem[0];
        }
    }
}

// block_size(L), grid_size(N)
template <typename T>
__global__ void gpu_version3(const T *__restrict__ input,
                             T *__restrict__ output,
                             const T *__restrict__ matrix, const int L,
                             const int M, const int N)
{
    // parallelize threadIdx.x over vector length, and blockIdx.x across k (N)
    // do initial vector reduction via warp-stride loop
    __shared__ T smem[my_L];
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int id = idy * warpSize + idx;
    int k = blockIdx.x;
    T v1;
    // vertical block-stride loop
    for (int y = threadIdx.y; y < L; y += blockDim.y)
    {
        v1 = 0;
        // horizontal warp-stride loop
        for (int x = threadIdx.x; x < M; x += warpSize)
        {
            v1 += input[k * M * L + y * M + x];
        }

        // warp-shuffle reduction
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {
            v1 += __shfl_down_sync(0xFFFFFFFF, v1, offset);
        }

        if (!threadIdx.x)
        {
            smem[y] = v1 / M;
        }
    }
    __syncthreads();
    v1 = smem[id];
    // matrix-vector multiply
    for (int i = 0; i < L; i++)
    {
        __syncthreads();
        smem[id] = v1 * matrix[i * L + id];
        for (int s = (blockDim.x * blockDim.y) >> 1; s > 0; s >>= 1)
        {
            __syncthreads();
            if (id < s)
            {
                smem[id] += smem[id + s];
            }
        }
        if (!id)
        {
            output[k + i * N] = smem[0];
        }
    }
}

template <typename T>
__global__ void gpu_version4(const T *__restrict__ input,
                             T *__restrict__ output,
                             const T *__restrict__ matrix, const int L,
                             const int M, const int N) {
  // parallelize threadIdx.x over vector length, and blockIdx.x across k (N)
  // do initial vector reduction via warp-stride loop
  __shared__ T smem[my_L];
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int id = idy * warpSize + idx;
  int k = blockIdx.x;
  T v1;
  // vertical block-stride loop
  for (int y = threadIdx.y; y < L; y += blockDim.y) {
    v1 = 0;
    // horizontal warp-stride loop
    for (int x = threadIdx.x; x < M; x += warpSize) {
      v1 += input[k * M * L + y * M + x];
    }

    // warp-shuffle reduction
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      v1 += __shfl_down_sync(0xFFFFFFFF, v1, offset);
    }

    if (!threadIdx.x) {
      smem[y] = v1 / M;
    }
  }
  __syncthreads();
  v1 = smem[id];
  for (int i = 0; i < L; i++) {  // matrix-vector multiply
    T v2 = v1 * matrix[i * L + id];
    // 1st warp-shuffle reduction
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
      v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
    }

    if (idx == 0) {
      smem[idy] = v2;
    }
    __syncthreads();  // put warp results in shared mem
    // hereafter, just warp 0
    if (idy == 0) {
      // reload v2 from shared mem if warp existed
      v2 = (idx < ((blockDim.x * blockDim.y) >> 5)) ? smem[idx] : 0;
      // final warp-shuffle reduction
      for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
    }
    if (!id) {
      output[k + i * N] = v2;
    }
  }
}

int main()
{
    const size_t input_bytes =
        static_cast<size_t>(my_N) * my_L * my_M * sizeof(float);
    const size_t matrix_bytes =
        static_cast<size_t>(my_L) * my_L * sizeof(float);
    const size_t output_bytes =
        static_cast<size_t>(my_L) * my_N * sizeof(float);

    float *input = nullptr;
    float *matrix = nullptr;
    float *output = nullptr;

    cudaMalloc(&input, input_bytes);
    cudaMalloc(&matrix, matrix_bytes);
    cudaMalloc(&output, output_bytes);

    cudaMemset(input, 0, input_bytes);
    cudaMemset(matrix, 0, matrix_bytes);

    // std::printf("gpu_version1<<<1, %d>>> L=%d M=%d N=%d\n", my_L, my_L,
    //             my_M, my_N);
    // gpu_version1<float><<<1, my_L>>>(input, output, matrix, my_L, my_M, my_N);
    // std::printf("gpu_version2<<<%d, %d>>> L=%d M=%d N=%d\n", my_N, my_L, my_L, my_M, my_N);
    // gpu_version2<float><<<my_N, my_L>>>(input, output, matrix, my_L, my_M, my_N);
    // std::printf("gpu_version3<<<%d, %d>>> L=%d M=%d N=%d\n", my_N, my_L, my_L, my_M, my_N);
    // gpu_version3<float><<<my_N, my_L>>>(input, output, matrix, my_L, my_M, my_N);
    std::printf("gpu_version4<<<%d, %d>>> L=%d M=%d N=%d\n", my_N, my_L, my_L, my_M, my_N);
    dim3 block(32, 32);
    gpu_version4<float><<<my_N, block>>>(input, output, matrix, my_L, my_M, my_N);
    cudaDeviceSynchronize();

    cudaFree(input);
    cudaFree(matrix);
    cudaFree(output);
    return 0;
}
