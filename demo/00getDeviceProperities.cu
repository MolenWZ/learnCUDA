#include<stdio.h>
#include<cuda_runtime.h>

/*
其他 deviceProp: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
*/
int main(int argc,char **argv){
    int deviceCount=0;
    cudaGetDeviceCount(&deviceCount);//获取当前可用的CUDA设备数量

    if (deviceCount == 0){
        printf("There are no available device(s) that support CUDA\n");
    }
    else{
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    //dev表示要查询的设备ID（设为0），driverVersion和runtimeVersion用于存储CUDA驱动程序版本和运行时版本。
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);//设置要使用的CUDA设备为dev（设备0）
    cudaDeviceProp deviceProp;//声明cudaDeviceProp结构体变量deviceProp，用于存储设备的属性。
    cudaGetDeviceProperties(&deviceProp, dev);//获取设备的详细属性，并将其存储在deviceProp中。
    printf("Device %d: \"%s\"\n\n", dev, deviceProp.name);//输出设备编号和设备名称

    cudaDriverGetVersion(&driverVersion);//获取CUDA驱动程序版本
    cudaRuntimeGetVersion(&runtimeVersion);//获取CUDA运行时版本
    
    printf("CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    //输出设备的CUDA（Compute Capability）主版本号和次版本号。
    printf("CUDA Capability Major/Minor version number:    %d.%d\n",deviceProp.major, deviceProp.minor);
    //设备的全局内存总量（以GB和字节为单位）
    printf("Total amount of global memory:                 %.2f GBytes (%llu bytes)\n", 
            (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
            (unsigned long long)deviceProp.totalGlobalMem);
    //GPU时钟频率（以MHz和GHz为单位）
    printf("GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f,deviceProp.clockRate * 1e-6f);
    //设备的内存时钟频率（以MHz为单位）和内存总线宽度（以位为单位）
    printf("Memory Clock rate:                             %.0f Mhz\n",deviceProp.memoryClockRate * 1e-3f);
    printf("Memory Bus Width:                              %d-bit\n",deviceProp.memoryBusWidth);
    //如果设备有L2缓存，打印L2缓存的大小
    if (deviceProp.l2CacheSize){
        printf("L2 Cache Size:                                 %d bytes\n",deviceProp.l2CacheSize);
    }
    //设备支持的最大纹理尺寸：1D、2D和3D
    printf("Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n", 
            deviceProp.maxTexture1D,
            deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
            deviceProp.maxTexture3D[2]);
    //层纹理的最大尺寸和层数
    printf("Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n", 
            deviceProp.maxTexture1DLayered[0],
            deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
            deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
    //常量内存总量、每个块的共享内存、每个块的可用寄存器数量和warp大小
    printf("Total amount of constant memory:               %zu bytes\n",deviceProp.totalConstMem);
    printf("Total amount of shared memory per block:       %zu bytes\n",deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block: %d\n",deviceProp.regsPerBlock);
    printf("Warp size:                                     %d\n",deviceProp.warpSize);
    //每个多处理器支持的最大线程数,共享内存和每个块的最大线程数
    printf("registers available per multiprocessor:        %u bytes\n",deviceProp.regsPerMultiprocessor);
    printf("Shared memory available per multiprocessor:    %zu bytes\n",deviceProp.sharedMemPerMultiprocessor);
    printf("Maximum number of threads per multiprocessor:  %d\n",deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of threads per block:           %d\n",deviceProp.maxThreadsPerBlock);
    //块和网格每个维度的最大尺寸
    printf("Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    //设备的最大内存填充（pitch）大小
    printf("Maximum memory pitch:                          %zu bytes\n",deviceProp.memPitch);
    
    exit(EXIT_SUCCESS);
}