#include <cuda_runtime.h>
#include <stdio.h>
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx,
                     const int ny){
    // 用于每行起始地址
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++){
        for (int ix = 0; ix < nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
    return;
}

__global__ void sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx,int ny){
    //数据总量为nxy=nx 乘 ny，block大小为32，grid大小为1
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;//全局线程索引

    if (ix < nx ) //线程约束，执行0到nx线程
        for (int iy = 0; iy < ny; iy++){ //每个线程需要执行一个循环，循环长度为ny
            int idx = iy * nx + ix; //线程索引
            MatC[idx] = MatA[idx] + MatB[idx];
        }
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // setp1: 设置GPU设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);//设备信息
    cudaSetDevice(dev);

    // setp2: 初始化矩阵
    int nx = 1 << 4; //矩阵列
    int ny = 1 << 4; //矩阵行
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    //分配主机内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // 在host上初始化数据
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 在host上计算矩阵相加
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    // setp4: 分配GPU内存
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);
    // 将数据从host传入设备上
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // 核函数，grid核block都设置为1维
    int dimx = 32;
    dim3 block(dimx, 1);//每个block，32个线程
    dim3 grid((nx + block.x - 1) / block.x, 1);
    sumMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    // setp6: 在主机中获取计算结果
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // 检查结果
    checkResult(hostRef, gpuRef, nxy);

    // 释放设备全局内存
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // 释放host内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // 重置设备
    cudaDeviceReset();

    return 0;

}