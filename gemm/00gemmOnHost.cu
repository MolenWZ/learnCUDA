#include <time.h>
#include <stdlib.h>
#include <iostream>

// 矩阵乘法函数1*A*B+1*C, shape: MK x NK = MN  
void gemmOnHost(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < N; ++j){
            for (int k = 0; k < K; ++k){
                C[i * N + j] += A[i * K + k] * B[j * K + k];
            }
        }
    }
}

void initialData(float *ip,int size){
    time_t  t;
    srand((unsigned) time(&t));
    for (int i=0;i<size;++i){
        ip[i] =(float)(rand() & 0xFF)/10.0f;
        //ip[i]=1.0f;
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

int main() {
    // 矩阵维度
    int M = 64; // A 的行数和 C 的行数
    int N = 64; // B 的列数和 C 的列数
    int K = 64; // A 的列数和 B 的行数
    size_t A_size = M*K;
    size_t B_size = K*N;
    size_t C_size = M*N;

    float *A, *B, *C;
    A = (float *)malloc(sizeof(float)*A_size);
    B = (float *)malloc(sizeof(float)*B_size);
    C = (float *)malloc(sizeof(float)*C_size);

    // 初始化矩阵 A, B 和 C
    initialData(A,A_size);
    initialData(B,B_size);
    memset(C, 0, C_size*sizeof(float));
    // initialData(C,C_size);

    //执行矩阵乘法
    gemmOnHost(M,N,K,A,B,C);

    // 打印结果矩阵 C
    std::cout << "Matrix C:" << std::endl;
    viewMat(C,M,N);

    free(A);
    free(B);
    free(C);
    return 0;
}