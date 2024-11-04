#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// 行优先
__global__ void BankConflictRowMajor(const int M, const int N, float *A)
{
    __shared__ float shareA[32][6]; // 共享内存声明
    for (int i = 0; i < N; i++)     // 从global内存读取数据到share内存
    {
        shareA[threadIdx.x][i] = A[threadIdx.x * N + i];
    }
    for (int i = 0; i < N; i++) // 使用share内存的数据进行运算
    {
        shareA[threadIdx.x][i] = shareA[threadIdx.x][i] * 2;
    }
    for (int i = 0; i < N; i++)
    {
        A[threadIdx.x * N + i] = shareA[threadIdx.x][i];
    }
}
// 行优先 padding
__global__ void BankConflictRowMajorPadding(const int M, const int N, float *A)
{
    __shared__ float shareA[32][6 + 1]; // 共享内存声明 padding 1 
    for (int i = 0; i < N; i++)     // 从global内存读取数据到share内存
    {
        shareA[threadIdx.x][i] = A[threadIdx.x * N + i];
    }
    for (int i = 0; i < N; i++) // 使用share内存的数据进行运算
    {
        shareA[threadIdx.x][i] = shareA[threadIdx.x][i] * 2;
    }
    for (int i = 0; i < N; i++)
    {
        A[threadIdx.x * N + i] = shareA[threadIdx.x][i];
    }
}
// 列优先
__global__ void BankConflictColMajor(const int M, const int N, float *A)
{
    __shared__ float shareA[6][32]; // 共享内存声明
    for (int i = 0; i < N; i++)     // 从global内存读取数据到share内存
    {
        shareA[i][threadIdx.x] = A[threadIdx.x * N + i];
    }
    for (int i = 0; i < N; i++) // 使用share内存的数据进行运算
    {
        shareA[i][threadIdx.x] = shareA[i][threadIdx.x] * 2;
    }
    for (int i = 0; i < N; i++)
    {
        A[threadIdx.x * N + i] = shareA[i][threadIdx.x];
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
    printf("%s Starting...\n", argv[0]);

    // setp1: 设置GPU设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name); // 设备信息
    cudaSetDevice(dev);

    // setp2: 初始化矩阵
    int M = 32;
    int N = 6;
    size_t nbytes = sizeof(float) * M * N;

    // 分配主机内存
    float *A;
    A = (float *)malloc(nbytes);

    // 在host上初始化数据
    for (int i = 0; i < M * N; ++i)
    {
        A[i] = i % 6;
    }

    std::cout << "Matrix A:" << std::endl;
    viewMat(A, M, N);

    // setp4: 分配GPU内存
    float *MatA;
    cudaMalloc((void **)&MatA, nbytes);
    // 将数据从host传入设备上
    cudaMemcpy(MatA, A, nbytes, cudaMemcpyHostToDevice);

    dim3 block(M);
    dim3 grid(1);
    BankConflictRowMajorPadding<<<grid, block>>>(M, N, MatA);
    cudaDeviceSynchronize();

    // setp6: 在主机中获取计算结果
    cudaMemcpy(A, MatA, nbytes, cudaMemcpyDeviceToHost);

    // 打印结果矩阵 C
    std::cout << "Result matrix A:" << std::endl;
    viewMat(A, M, N);

    // 释放设备全局内存
    cudaFree(MatA);
    // 释放host内存
    free(A);
    // 重置设备
    cudaDeviceReset();

    return 0;
}