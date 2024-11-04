#include <cuda_runtime.h>
#include <iostream>
// #define TINY //如果是小矩阵，打印出来

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; ++i)
    {
        // ip[i] =(float)(rand() & 0xFF)/10.0f;
        ip[i] = 1.0f;
    }
}

__global__ void MemoryUncoalescing(const int M, float *A, float *B, float *C)
{
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread < M)
    {
        for (int j = 0; j < M; j++)
        {
            float Pvalue = 0;
            for (int k = 0; k < M; k++)
            {
                Pvalue += A[thread * M + k] * B[thread * M + j];
            }
            C[thread * M + j] = Pvalue;
        }
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
    std::cout << argv[0] << " Starting..." << std::endl;

    // setp1: 设置GPU设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name); // 设备信息
    cudaSetDevice(dev);

    // setp2: 初始化矩阵
    int K = 4;

    size_t nbytes = sizeof(float) * K * K;

    // 分配主机内存
    float *M, *N, *C;
    M = (float *)malloc(nbytes);
    N = (float *)malloc(nbytes);
    C = (float *)malloc(nbytes);

    // 在host上初始化数据
    for (int i = 0; i < K * K; ++i)
    {
        M[i] = i % 4;
        N[i] = i % 4+1;
        C[i] = 0;
    }
    std::cout << "Matrix M:" << std::endl;
    viewMat(M, K, K);
    std::cout << "Matrix N:" << std::endl;
    viewMat(N, K, K);
    std::cout << "Matrix C:" << std::endl;
    viewMat(C, K, K);

    // setp4: 分配GPU内存，将数据从host传入设备上
    float *MatM, *MatN, *MatC;
    cudaMalloc((void **)&MatM, nbytes);
    cudaMalloc((void **)&MatN, nbytes);
    cudaMalloc((void **)&MatC, nbytes);
    cudaMemcpy(MatM, M, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(MatN, N, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(MatC, C, nbytes, cudaMemcpyHostToDevice);

    // 核函数，grid核block都设置为1维
    dim3 block(4); // 每个block，4个线程
    dim3 grid((K + block.x - 1) / block.x);
    MemoryUncoalescing<<<grid, block>>>(K, MatM, MatN, MatC);
    cudaDeviceSynchronize();

    // setp6: 在主机中获取计算结果
    cudaMemcpy(C, MatC, nbytes, cudaMemcpyDeviceToHost);

    // 打印结果矩阵 C
    std::cout << "Result matrix C:" << std::endl;
    viewMat(C, K, K);

    // 释放设备全局内存
    cudaFree(MatM);
    cudaFree(MatN);

    // 释放host内存
    free(M);
    free(N);

    // 重置设备
    cudaDeviceReset();

    return 0;
}