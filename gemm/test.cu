#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "general.h"
#include <cublas_v2.h>
/*
编译： nvcc test.cu -lcublas -o test
*/
__global__ void gemmNaiveOnGPU(const int M, const int N, const int K,
                               const float *A, float alpha, const float *B, float beta, float *C)
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

void gemmNaive(const int M, const int N, const int K,
               const float *A, float alpha, const float *B, float beta, float *C)
{
    dim3 block(32, 32); // 每个block，32x32个线程
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemmNaiveOnGPU<<<grid, block>>>(M, N, K, A, alpha, B, beta, C);
}

template <const int BM, const int BK, const int BN>
__global__ void gemmSharedMemOnGPU(const int M, const int N, const int K,
                                   const float *A, float alpha, const float *B, float beta, float *C)
{
    // const int BM = 32;
    // const int BK = 32;
    // const int BN = 32;
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

void gemmSharedMem(const int M, const int N, const int K,
                   const float *A, float alpha, const float *B, float beta, float *C)
{

    const int BM = 32;
    const int BK = 32;
    const int BN = 32;
    dim3 block(BN, BM); // 每个block，32个线程
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemmSharedMemOnGPU<BM, BK, BN><<<grid, block>>>(M, N, K, A, alpha, B, beta, C);
}

template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void gemmThreadTilingOnGPU(const int M, const int N, const int K,
                                      const float *A, float alpha, const float *B, float beta, float *C)
{
    // 共享内存行列设置，共享内存声明
    // const int BM = 32;
    // const int BK = 16;
    // const int BN = 32;
    __shared__ float shareA[BM * BK];
    __shared__ float shareB[BK * BN];

    // 用 register 行列设置，储存计算结果. 寄存器变量声明
    // const int TM = 8;
    // const int TN = 4;
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
void gemmThreadTiling(const int M, const int N, const int K,
                      const float *A, float alpha, const float *B, float beta, float *C)
{
    const int BM = 32;
    const int BK = 16;
    const int BN = 32;
    const int TM = 8;
    const int TN = 4;
    dim3 block(BM * BN / (TM * TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemmThreadTilingOnGPU<BM, BK, BN, TM, TN><<<grid, block>>>(M, N, K, A, alpha, B, beta, C);
}

template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void gemmVectorizedMemOnGPU(const int M, const int N, const int K,
                                       float *A, float alpha, float *B, float beta, float *C)
{
    // 共享内存设置，声明
    // const int BM = 32;
    // const int BK = 16;
    // const int BN = 32;
    __shared__ float shareA[BK][BM]; // 通过padding的方式避免bank conflict
    __shared__ float shareB[BK * BN];

    // 用 register 储存计算结果. 寄存器变量声明
    // const int TM = 8;
    // const int TN = 4;
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
    int strideA = threadNumsPerBlock / (BK / 4);  // 32 / (16/4)=8 BK条数据，每4个一组（vector），变成了BK/4条数据,相应的步长边长了4倍
    int strideB = threadNumsPerBlock / (BN / 4);  // 32 / (32/4)=4

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
void gemmVectorizedMem(const int M, const int N, const int K,
                       float *A, float alpha, float *B, float beta, float *C)
{
    const int BM = 64;
    const int BK = 16;
    const int BN = 64;
    const int TM = 8;
    const int TN = 4;
    dim3 block(BM * BN / (TM * TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemmVectorizedMemOnGPU<BM, BK, BN, TM, TN><<<grid, block>>>(M, N, K, A, alpha, B, beta, C);
}

// 测量函数执行时间的结构体
struct CudaEventPair
{
    cudaEvent_t start;
    cudaEvent_t end;

    CudaEventPair()
    {
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
    }

    ~CudaEventPair()
    {
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(end));
    }
};

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

    // 用于测量gemmNaive函数执行时间
    CudaEventPair gemmNaiveEvents;
    cudaEventRecord(gemmNaiveEvents.start, 0);
    gemmNaive(M, N, K, MatA, alpha, MatB, beta, MatC); // gemmNaiveOnGPU
    cudaEventRecord(gemmNaiveEvents.end, 0);
    CHECK(cudaEventSynchronize(gemmNaiveEvents.end));
    float gemmNaiveTime;
    cudaEventElapsedTime(&gemmNaiveTime, gemmNaiveEvents.start, gemmNaiveEvents.end);
    std::cout << "gemmNaive function execution time: " << gemmNaiveTime << " ms" << std::endl;

    // 用于测量gemmSharedMem函数执行时间
    CudaEventPair gemmSharedMemEvents;
    cudaEventRecord(gemmSharedMemEvents.start, 0);
    gemmSharedMem(M, N, K, MatA, alpha, MatB, beta, MatC); // gemmSharedMemOnGPU
    cudaEventRecord(gemmSharedMemEvents.end, 0);
    CHECK(cudaEventSynchronize(gemmSharedMemEvents.end));
    float gemmSharedMemTime;
    cudaEventElapsedTime(&gemmSharedMemTime, gemmSharedMemEvents.start, gemmSharedMemEvents.end);
    std::cout << "gemmSharedMem function execution time: " << gemmSharedMemTime << " ms" << std::endl;

    // 用于测量gemmThreadTiling函数执行时间
    CudaEventPair gemmThreadTilingEvents;
    cudaEventRecord(gemmThreadTilingEvents.start, 0);
    gemmThreadTiling(M, N, K, MatA, alpha, MatB, beta, MatC); // gemmThreadTilingOnGPU
    cudaEventRecord(gemmThreadTilingEvents.end, 0);
    CHECK(cudaEventSynchronize(gemmThreadTilingEvents.end));
    float gemmThreadTilingTime;
    cudaEventElapsedTime(&gemmThreadTilingTime, gemmThreadTilingEvents.start, gemmThreadTilingEvents.end);
    std::cout << "gemmThreadTiling function execution time: " << gemmThreadTilingTime << " ms" << std::endl;

    // 用于测量gemmVectorizedMem函数执行时间
    CudaEventPair gemmVectorizedMemEvents;
    cudaEventRecord(gemmVectorizedMemEvents.start, 0);
    gemmVectorizedMem(M, N, K, MatA, alpha, MatB, beta, MatC); // gemmVectorizedMemOnGPU
    cudaEventRecord(gemmVectorizedMemEvents.end, 0);
    CHECK(cudaEventSynchronize(gemmVectorizedMemEvents.end));
    float gemmVectorizedMemTime;
    cudaEventElapsedTime(&gemmVectorizedMemTime, gemmVectorizedMemEvents.start, gemmVectorizedMemEvents.end);
    std::cout << "gemmVectorizedMem function execution time: " << gemmVectorizedMemTime << " ms" << std::endl;

    // 用于测量cublasSgemm函数执行时间
    CudaEventPair cublasSgemmEvents;
    cublasHandle_t handle; // cublas
    cublasCreate(&handle);
    cudaEventRecord(cublasSgemmEvents.start, 0);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, MatB, N, MatA, K, &beta, MatC, N);
    cudaEventRecord(cublasSgemmEvents.end, 0);
    CHECK(cudaEventSynchronize(cublasSgemmEvents.end));
    float cublasSgemmTime;
    cudaEventElapsedTime(&cublasSgemmTime, cublasSgemmEvents.start, cublasSgemmEvents.end);
    std::cout << "cublasSgemm function execution time: " << cublasSgemmTime << " ms" << std::endl;

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
    cublasDestroy(handle);
    // CHECK(cudaDeviceReset());
    return 0;
}