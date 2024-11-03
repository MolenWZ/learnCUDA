#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, const float ival, int size){
    for (int i = 0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }
    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
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

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // set up data size of matrix
    int nx = 1 << 8;
    int ny = 1 << 8;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A,  2.0f, nxy);
    initialData(h_B,  0.5f, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
    
    // set block size -------------------------------------------------------
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy); // block size:(32,32)
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    // adjust block size -------------------------------------------------------
    block.x = 16;   // block size:(16,32)
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    // adjust block size--------------------------------------------------------
    block.y = 16;
    block.x = 32;  // block size:(32,16)
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    // adjust block size--------------------------------------------------------
    block.y = 16;
    block.x = 16; // block size:(16,16)
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    // adjust block size--------------------------------------------------------
    block.y = 16;
    block.x = 64; // block size:(64,16)
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    // adjust block size--------------------------------------------------------
    block.y = 64;
    block.x = 16; // block size:(16,64)
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;

}