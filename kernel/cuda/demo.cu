// Build: /usr/local/cuda-12.8/bin/nvcc -O2 -std=c++17 -arch=sm_89 \
//        demo.cu -o demo
// Run:   ./demo

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(error));                   \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

__global__ void matrix_add(const float* a, const float* b, float* c, int rows,
                           int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        const int i = row * cols + col;
        c[i] = a[i] + b[i];
    }
}

int main()
{
    constexpr int rows = 23;
    constexpr int cols = 37;
    const size_t count = static_cast<size_t>(rows) * cols;
    const size_t bytes = count * sizeof(float);

    std::vector<float> h_a(count);
    std::vector<float> h_b(count);
    std::vector<float> h_c(count);
    for (size_t i = 0; i < count; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);

    constexpr int warmup_iterations = 10;
    constexpr int timed_iterations = 1000;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup_iterations; ++i) {
        matrix_add<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iterations; ++i) {
        matrix_add<<<grid, block>>>(d_a, d_b, d_c, rows, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int row = 0; row < rows && passed; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int i = row * cols + col;
            const float expected = h_a[i] + h_b[i];
            if (std::fabs(h_c[i] - expected) > 1e-6f) {
                std::fprintf(stderr,
                             "FAIL at (%d, %d): got %.6f, expected %.6f\n",
                             row, col, h_c[i], expected);
                passed = false;
                break;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    if (passed) {
        std::printf(
            "PASS: matrix_add<<<grid=(%u,%u), block=(%u,%u)>>> checked %dx%d\n",
            grid.x, grid.y, block.x, block.y, rows, cols);
        std::printf("CUDA Event: %d launches, total %.3f ms, average %.3f us\n",
                    timed_iterations, elapsed_ms,
                    elapsed_ms * 1000.0f / timed_iterations);
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
