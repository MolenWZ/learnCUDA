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
    float *A,
    float alpha,
    float *B,
    float beta,
    float *C)
{
    // 共享内存设置，声明
    const int BM = 32;
    const int BK = 16;
    const int BN = 32;
    __shared__ float shareA[BK][BM]; // 通过padding的方式避免bank conflict
    __shared__ float shareB[BK * BN];

    // 用 register 储存计算结果. 寄存器变量声明
    const int TM = 8;
    const int TN = 4;
    float threadResults[TM * TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    // 确定线程负责的结果矩阵 C 的索引
    const unsigned int threadRow = (threadIdx.x * TN / BN) * TM;
    const unsigned int threadCol = threadIdx.x * TN % BN;

    // 全局内存指针偏移
    A += blockIdx.x * BM * K;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * N + blockIdx.y * BN;

    // 计算线程块内线程数量及相关步长
    int threadNumsPerBlock = BM * BN / (TM * TN); // 32*32/(8*4)=32
    int strideA = threadNumsPerBlock / (BK / 4); // 32 / (16/4)=8 BK条数据，每4个一组（vector），变成了BK/4条数据,相应的步长边长了4倍
    int strideB = threadNumsPerBlock / (BN / 4); // 32 / (32/4)=4

    // 确定加载到共享内存的索引
    const unsigned int innerRowA = threadIdx.x / (BK / 4); // 0:1:31 / (16/4) = 0#4:1:7#4 对应BK,BN都缩减4倍数，innerRowA最大增加了4倍
    const unsigned int innerColA = threadIdx.x % (BK / 4); // 0:1:31 % (16/4) = 0:1:3#4 BK列，给线程分行
    const unsigned int innerRowB = threadIdx.x / (BN / 4); // 0:1:31 / (32/4) = 0#8:1:3#8
    const unsigned int innerColB = threadIdx.x % (BN / 4); // 0:1:31 % (32/4) = 0:1:7#4

    // 循环加载数据到共享内存并同步
    for (int tile_idx = 0; tile_idx < K; tile_idx += BK) // 外循环，点积运算累加
    {
        for (int loadoffset = 0; loadoffset < BM; loadoffset += strideA)
        {  
            // 声明4倍float长度的变量tmp并强制转换,float4是结构体，通过x,y,z,w依次访问第1,2,3,4个元素。
            float4 tmp = reinterpret_cast<float4 *>(&A[(innerRowA + loadoffset) * K + innerColA * 4])[0]; // 第0个float4，实际上也只有1个
            shareA[(innerColA * 4 + 0)][innerRowA + loadoffset] = tmp.x;
            shareA[(innerColA * 4 + 1)][innerRowA + loadoffset] = tmp.y;
            shareA[(innerColA * 4 + 2)][innerRowA + loadoffset] = tmp.z;
            shareA[(innerColA * 4 + 3)][innerRowA + loadoffset] = tmp.w;
        }
        for (int loadoffset = 0; loadoffset < BK; loadoffset += strideB)
        {
            reinterpret_cast<float4 *>(&shareB[(innerRowB + loadoffset) * BN + innerColB * 4])[0] =
                reinterpret_cast<float4 *>(&B[(innerRowB + loadoffset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++)
        {
            // 导入寄存器
            for (int i = 0; i < TM; i++)
            {
                regA[i] = shareA[dotIdx][(threadRow + i)];
            }
            for (int i = 0; i < TN; i += 4)
            {
                reinterpret_cast<float4 *>(&regB[i])[0] =
                    reinterpret_cast<float4 *>(&shareB[dotIdx * BN + threadCol + i])[0];
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
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM)
    {
        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow + resIdxM) * N + threadCol + resIdxN])[0];
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
            reinterpret_cast<float4 *>(
                &C[(threadRow + resIdxM) * M + threadCol + resIdxN])[0] = tmp;
        }
    }
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
    // initialData(C, M * N);

    // std::cout << "Matrix A:" << std::endl;
    // viewMat(A, M, K);
    // std::cout << "Matrix B:" << std::endl;
    // viewMat(B, K, N);
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
    dim3 block(dimy * dimx / (8 * 4)); // 每个block，32个线程
    dim3 grid((N + dimx - 1) / dimx, (M + dimy - 1) / dimy);
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