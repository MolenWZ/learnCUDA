/*
Vectorized Memory Access

https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
*/

#include <cuda_runtime.h>
#define MAX_BLOCKS 1024

/*--------------------           scalar          -----------------*/
__global__ void device_copy_scalar_kernel(int *d_in, int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        d_out[i] = d_in[i];
    }
}

void device_copy_scalar(int *d_in, int *d_out, int N)
{
    int threads = 128;
    int blocks = min((N + threads - 1) / threads, MAX_BLOCKS);
    device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N);
}

/*--------------------           vector2          -----------------*/
__global__ void device_copy_vector2_kernel(int *d_in, int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x)
    {
        reinterpret_cast<int2 *>(d_out)[i] = reinterpret_cast<int2 *>(d_in)[i];
    }

    // in only one thread, process final element (if there is one)
    if (idx == N / 2 && N % 2 == 1)
        d_out[N - 1] = d_in[N - 1];
}

void device_copy_vector2(int *d_in, int *d_out, int N)
{
    int threads = 128;
    int blocks = min((N / 2 + threads - 1) / threads, MAX_BLOCKS);
    device_copy_vector2_kernel<<<blocks, threads>>>(d_in, d_out, N);
}

/*--------------------           vector4          -----------------*/
__global__ void device_copy_vector4_kernel(int *d_in, int *d_out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 4; i += blockDim.x * gridDim.x)
    {
        reinterpret_cast<int4 *>(d_out)[i] = reinterpret_cast<int4 *>(d_in)[i];
    }

    // in only one thread, process final elements (if there are any)
    int remainder = N % 4;
    if (idx == N / 4 && remainder != 0)
    {
        while (remainder)
        {
            int idx = N - remainder--;
            d_out[idx] = d_in[idx];
        }
    }
}

void device_copy_vector4(int *d_in, int *d_out, int N)
{
    int threads = 128;
    int blocks = min((N / 4 + threads - 1) / threads, MAX_BLOCKS);
    device_copy_vector4_kernel<<<blocks, threads>>>(d_in, d_out, N);
}

int main()
{
    int size = 10240;
    int *d_in, *d_out;

    size_t nbytes = sizeof(int) * size;

    cudaMalloc((void **)&d_in, nbytes);
    cudaMalloc((void **)&d_out, nbytes);
    device_copy_scalar(d_in, d_out, size);
    device_copy_vector2(d_in, d_out, size);
    device_copy_vector4(d_in, d_out, size);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}