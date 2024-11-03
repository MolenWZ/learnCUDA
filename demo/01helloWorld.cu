#include<stdio.h>

__global__ void helloFromGPU(){
    printf("Hello from GPU,threadId:(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
    printf("Hello from CPU\n");
    helloFromGPU<<<1,5>>>();
    cudaDeviceReset();
    return 0;
}