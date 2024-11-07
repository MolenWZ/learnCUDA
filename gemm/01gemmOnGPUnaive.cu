#include <cuda_runtime.h>
#include <iostream>
#include "general.h"

__global__ void gemmNaiveOnGPU(
    const int M,
    const int N,
    const int K,
    const float *A,
    float alpha,
    const float *B,
    float beta,
    float *C)
{
    // 不能内存合并
    const int cRow = blockIdx.x * blockDim.x + threadIdx.x;
    const int cCol = blockIdx.y * blockDim.y + threadIdx.y;
    // 内存合并
    // const int cCol = blockIdx.x * blockDim.x + threadIdx.x;
    // const int cRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (cRow < M && cCol < N) // Thread constraint
    {
        float tmp = 0.0;

        for (int k = 0; k < K; ++k)
        {
            tmp += A[cRow * K + k] * B[k * N + cCol];
        }
        // C = alpha*A*B+beta*C
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}

int main(int argc, char **argv)
{
    std::cout << argv[0] << " Starting..." << std::endl;

    // setp1: 设置GPU设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    std::cout << "Using Device " << dev << " : " << deviceProp.name << std::endl; // 设备信息
    cudaSetDevice(dev);

    // setp2: 初始化矩阵
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t A_size = M * K;
    size_t B_size = K * N;
    // size_t C_size = M * N;
    size_t A_bytes = sizeof(float) * M * K;
    size_t B_bytes = sizeof(float) * K * N;
    size_t C_bytes = sizeof(float) * M * N;

    // 分配主机内存
    float *A, *B, *C;
    A = (float *)malloc(A_bytes);
    B = (float *)malloc(B_bytes);
    C = (float *)malloc(C_bytes);
    float alpha = 1.0;
    float beta = 1.0;

    // 在host上初始化数据
    initialData(A, A_size);
    initialData(B, B_size);
    memset(C, 0, C_bytes);
    // initialData(C, C_size);  // 或则随机初始化C

    // std::cout << "Matrix A:" << std::endl;
    // viewMat(A, M, K);
    // std::cout << "Matrix B:" << std::endl;
    // viewMat(B, K, M);
    // std::cout << "Matrix C:" << std::endl;
    // viewMat(C, M, N);

    // setp4: 分配GPU内存，将数据从host传入设备上
    float *MatA, *MatB, *MatC;
    CHECK(cudaMalloc((void **)&MatA, A_bytes));
    CHECK(cudaMalloc((void **)&MatB, B_bytes));
    CHECK(cudaMalloc((void **)&MatC, C_bytes));
    CHECK(cudaMemcpy(MatA, A, A_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MatB, B, B_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MatC, C, C_bytes, cudaMemcpyHostToDevice));

    // 核函数，grid核block都设置为1维
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy); // 每个block，32个线程
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemmNaiveOnGPU<<<grid, block>>>(M, N, K, MatA, alpha, MatB, beta, MatC);

    CHECK(cudaDeviceSynchronize());

    // setp6: 在主机中获取计算结果
    CHECK(cudaMemcpy(C, MatC, C_bytes, cudaMemcpyDeviceToHost));

    // 打印结果矩阵 C

    // std::cout << "Result matrix C:" << std::endl;
    // viewMat(C, M, N);

    // 释放设备全局内存
    CHECK(cudaFree(MatA));
    CHECK(cudaFree(MatB));
    CHECK(cudaFree(MatC));

    // 释放host内存
    free(A);
    free(B);
    free(C);

    // 重置设备
    CHECK(cudaDeviceReset());

    return 0;
}