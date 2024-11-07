#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "general.h"
// 0#4 表示4个0，即{0,0,0,0}
// 0:1:3 表示从0到3，step为1，即{0,1,2,3}
// 0:1:3#2 表示2个0:1:3，即{0,1,2,3,0,1,2,3}
// 0#2:1:3#2 表示2个从0到3，step为1，即{0,0,1,1,2,2,3,3}
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
    // 共享内存行列设置，共享内存声明
    const int BM = 32;
    const int BK = 16;
    const int BN = 32;
    __shared__ float shareA[BM * BK];
    __shared__ float shareB[BK * BN];

    // 用 register 行列设置，储存计算结果. 寄存器变量声明
    const int TM = 8;
    const int TN = 4;
    float threadResults[TM * TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    // 确定线程负责的结果，矩阵C tile的索引
    const int threadRow = (threadIdx.x * TN / BN) * TM; // (0:1:31 * 4 / 32) * 8 = (0:4:124 / 32) * 8 = 0#8:1:3#8
    const int threadCol = threadIdx.x * TN % BN;        // (0:1:31 * 4 % 32) = 0:4:124 % 32 = 0:1:31#4

    // 全局内存指针偏移得到block中的矩阵A,B,C的首地址
    A += blockIdx.x * BM * K;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * N + blockIdx.y * BN;

    /*
    索引设置：实现加载矩阵A,B到共享内存
    得到线程的数量threadNumsPerBlock，那么在复制A,B数据到共享内存时，每次迭代转移threadNumsPerBlock条数据，
    对于下一次迭代，矩阵A的步长就是 threadNumsPerBlock/BK
    对于矩阵B(数据纵向导入),每次加载threadNumsPerBlock条数据，对于纵向下一次迭代，步长为threadNumsPerBlock/BN
    */
    int threadNumsPerBlock = BM * BN / (TM * TN); // 32*32/(8*4) = 32
    int strideA = threadNumsPerBlock / BK;        // 32 / 16 = 2
    int strideB = threadNumsPerBlock / BN;        // 32 /32 = 1
    // 确定加载到共享内存的索引，线程thred(0,end) 将矩阵A reshape 成BK列，（-1，BK）,矩阵B是shape:(BN,-1)
    const int innerRowA = threadIdx.x / BK; // 0:1:31 / 16 = 0#16:1:1#16  共享内存BK列，给线程分行
    const int innerColA = threadIdx.x % BK; // 0:1:31 % 16 = 0:1:15#2 BK列，给线程分行
    const int innerRowB = threadIdx.x / BN; // 0:1:31 / 32 = 0#32
    const int innerColB = threadIdx.x % BN; // 0:1:31 % 32 = 0:1:31

    // 循环加载数据到共享内存并同步
    for (int tile_idx = 0; tile_idx < K; tile_idx += BK)
    {
        for (int loadoffset = 0; loadoffset < BM; loadoffset += strideA)
        {
            shareA[(innerRowA + loadoffset) * BK + innerColA] = A[(innerRowA + loadoffset) * K + innerColA];
        }
        for (int loadoffset = 0; loadoffset < BK; loadoffset += strideB)
        {
            shareB[(innerRowB + loadoffset) * BN + innerColB] = B[(innerRowB + loadoffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++)
        {
            // 导入寄存器
            for (int i = 0; i < TM; i++)
            {
                regA[i] = shareA[(threadRow + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; i++)
            {
                regB[i] = shareB[dotIdx * BN + threadCol + i];
            }
            for (int resIdxM = 0; resIdxM < TM; resIdxM++)
            {
                for (int resIdxN = 0; resIdxN < TN; resIdxN++)
                {
                    threadResults[resIdxM * TN + resIdxN] += regA[resIdxM] * regB[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    // 将计算结果写回全局内存
    for (int resIdxM = 0; resIdxM < TM; resIdxM++)
    {
        for (int resIdxN = 0; resIdxN < TN; resIdxN++)
        {
            C[(threadRow + resIdxM) * N + threadCol + resIdxN] = threadResults[resIdxM * TN + resIdxN];
        }
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
    CHECK(cudaSetDevice(dev));

    // setp2: 初始化矩阵
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
    // initialData(C, M * N);

    // std::cout << "Matrix A:" << std::endl;
    // viewMat(A, M, K);
    // std::cout << "Matrix B:" << std::endl;
    // viewMat(B, K, N);
    // std::cout << "Matrix C:" << std::endl;
    // viewMat(C, M, N);

    // setp4: 分配GPU内存
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
    dim3 block(dimy * dimx / (8 * 4));
    dim3 grid((N + dimx - 1) / dimx, (M + dimy - 1) / dimy);
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