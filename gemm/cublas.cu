#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include "general.h"
// nvcc 000cublas.cu -lcublas
// https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api
/*
cuBLAS 库公开了三组 API：
cuBLAS API，在本文档中简称为 cuBLAS API（从 CUDA 6.0 开始），
cuBLASXt API（从 CUDA 6.0 开始），以及
cuBLASLt API（从 CUDA 10.1 开始）
*/

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
    // viewMat(B, N, K);
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
    cublasHandle_t handle;
    cublasCreate(&handle);
    // if(cublasCreate(&handle)){
    //     std::cerr << "Create cublas handle error." << std::endl;
    //     exit(EXIT_FAILURE);
    // };
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, MatB, N, MatA, K, &beta, MatC, N);
    
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
    CHECK(cudaDeviceReset());
    return 0;
}