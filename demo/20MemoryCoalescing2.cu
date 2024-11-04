#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

__global__ void MemoryUncoalescing(const int S, float *Mat)
{
    for (int s = 0; s < S; s++)
    {
        printf("%f ", Mat[threadIdx.x * S + s]);
    }
}

__global__ void MemoryCoalescing(const int K, float *Mat)
{
    for (int k = 0; k < K; k++)
    {
        printf("%f ", Mat[k * K + threadIdx.x]);
    }
}

void viewMat(const float *p, const int r, const int c)
{
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < c; ++j)
        {
            std::cout << p[i * c + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    int K = 64;
    int S = 4;
    size_t nbytes = sizeof(float) * K * S;

    float *M;
    M = (float *)malloc(nbytes);
    
    for (int i = 0; i < K * S; ++i)
    {
        M[i] = i % S;
    }

    std::cout << "Matrix M:" << std::endl;
    viewMat(M, K, S);

    float *MatM;
    cudaMalloc((void **)&MatM, nbytes);
    cudaMemcpy(MatM, M, nbytes, cudaMemcpyHostToDevice);
    dim3 block(4); // 1个block，4个线程
    dim3 grid(1);
    //MemoryUncoalescing<<<grid, block>>>(S, MatM);
    MemoryCoalescing<<<grid, block>>>(K, MatM);

    cudaFree(MatM);
    free(M);
    return 0;
}