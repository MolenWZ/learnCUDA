#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void)
{
    // gird维度(block的排列)，block维度（thread的排列）
    //打印block的索引，和该block中的线程索引
    printf("gridDim:(%d, %d, %d) blockDim:(%d, %d, %d) blockIdx:(%d, %d, %d) threadIdx:(%d, %d, %d)\n", 
        gridDim.x, gridDim.y, gridDim.z, 
        blockDim.x, blockDim.y, blockDim.z, 
        blockIdx.x, blockIdx.y, blockIdx.z, 
        threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv)
{
    // 数据大小
    int nElem = 6;

    // block排列和线程排列
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x); //等同于向上取整，得到grid的大小

    // 从主机侧检查网格和块尺寸
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // 从设备侧检查网格和块尺寸
    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

    return(0);
}