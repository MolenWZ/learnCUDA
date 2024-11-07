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
    const int BM = 32;
    const int BK = 32;
    const int BN = 32;
    // 共享内存声明
    __shared__ float shareA[BM][BK];
    __shared__ float shareB[BK][BN];

    // 全局内存指针偏移
    A += blockIdx.x * BM * K;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * N + blockIdx.y * BN;

    // 循环加载数据到共享内存并同步
    float tmp = 0.0;
    for (int i = 0; i < K / BK; ++i) // 外层循环，循环次数为 k/BK
    {
        // 将会从global内存读取数据到share内存
        shareA[threadIdx.y][threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        shareB[threadIdx.y][threadIdx.x] = B[threadIdx.y * N + threadIdx.x];
        // 更新指针
        A += BK;
        B += BK * N;
        __syncthreads();             // 进行线程同步
        for (int j = 0; j < BK; ++j) // 内层循环，循环次数为 BK。在每次循环中进行乘法运算，然后累加到 tmp 上。
        {
            tmp += shareA[threadIdx.y][j] * shareB[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[threadIdx.y * N + threadIdx.x] = alpha * tmp + beta * C[threadIdx.y * N + threadIdx.x];
}

int main(int argc, char **argv)
{
    std::cout << argv[0] << " Starting..." << std::endl;

    // 设置GPU设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    std::cout << "Using Device " << dev << " : " << deviceProp.name << std::endl; // 设备信息
    CHECK(cudaSetDevice(dev));

    // 初始化矩阵
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t A_size = M * K;
    size_t B_size = K * N;
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

    // 分配GPU内存
    float *MatA, *MatB, *MatC;
    CHECK(cudaMalloc((void **)&MatA, A_bytes));
    CHECK(cudaMalloc((void **)&MatB, B_bytes));
    CHECK(cudaMalloc((void **)&MatC, C_bytes));
    // 将数据从host传入设备上
    CHECK(cudaMemcpy(MatA, A, A_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MatB, B, B_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(MatC, C, C_bytes, cudaMemcpyHostToDevice));

    // 核函数
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy); // 每个block，32个线程
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemmNaiveOnGPU<<<grid, block>>>(M, N, K, MatA, alpha, MatB, beta, MatC);

    CHECK(cudaDeviceSynchronize());
    // 在主机中获取计算结果
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