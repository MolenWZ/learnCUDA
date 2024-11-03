#include<stdio.h>

__global__ void helloFromGPU(){
    if (threadIdx.x<3)
    printf("Hello from GPU, above threadId:%d\n", threadIdx.x);
    if (threadIdx.x>=3)
    printf("Hello from GPU, under threadId:%d\n", threadIdx.x);
}

int main(){
    printf("Hello from CPU\n");
    helloFromGPU<<<1,5>>>();
    cudaDeviceReset();
    return 0;
}