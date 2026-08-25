#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//     Branch Efficiency                   %        74.05
//     Avg. Divergent Branches      branches         1.26
__global__ void SimpleReduce(float *in, float *out)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = tid * 2;
    // blockDim.x == N / 2, max global thread id = N / 2 - 1;
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        if (tid % stride == 0)
        {
            in[idx] += in[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        *out = in[0];
    }
}

//     Branch Efficiency                   %          100
//     Avg. Divergent Branches      branches            0
__global__ void SimpleReduceV2(float *in, float *out)
{
    unsigned int tid = threadIdx.x;
    for (int stride = blockDim.x; stride > 0; stride /= 2)
    {
        in[tid] += in[tid+stride];
        __syncthreads();
    }
    if (tid == 0)
    {
        *out = in[0];
    }
}

bool CheckResult(float *result, float groundTruth)
{
    if (*result != groundTruth)
    {
        return false;
    }
    return true;
}

int main()
{
    const int N = 2048;
    const int bytes = N * sizeof(float);

    // allocate host mem
    float *h_in = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(sizeof(float));

    // init host
    for (int i = 0; i < N; i++)
    {
        h_in[i] = 1.0f;
    }

    // allocate device mem
    float *d_in;
    float *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));

    // copy input from host to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // run kernel
    int grid_size = 1;
    int block_size = N / 2;
    SimpleReduceV2<<<grid_size, block_size>>>(d_in, d_out);

    // record time
    cudaEvent_t start, end;
    float timecost = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // copy result from device to host
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&timecost, start, end);
    std::cout << "Time Cost " << timecost << " ms" <<std::endl;

    // check result
    bool is_right = CheckResult(h_out, N);
    if (is_right)
    {
        std::cout << "Right Answer" << std::endl;
    }
    else
    {
        std::cout << "Wrong Answer " << *h_out << std::endl;
    }

    // free mem
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}