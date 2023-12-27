#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("There are %d device\n", deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        int driver_version, runtime_version;
        cudaGetDeviceProperties(&prop, dev);
        printf("Device%d:                                       %s\n",
               dev, prop.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA dirver veriosn:                            %d.%d\n",
               driver_version/1000, (driver_version % 1000)/10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA runtime veriosn:                           %d.%d\n",
               runtime_version/1000, (runtime_version % 1000)/10);
        printf("Device Prop:                                    %d.%d\n",
               prop.major, prop.minor);
        printf("Maximum Golbal Memory Size:                     %.2f GB\n",
               prop.totalGlobalMem*1.0/(1024*1024*1024));
        printf("Maximum Constant Memory Size:                   %ld KB\n",
               prop.totalConstMem/1024);
        printf("Maximum shared memory size per block:           %ld KB\n",
               prop.sharedMemPerBlock/1024);
        printf("Maximum block per SM:                           %d\n",
               prop.maxBlocksPerMultiProcessor);
        printf("Maximum registers per SM:                       %d\n",
               prop.regsPerBlock);
        printf("Maximum block dimensions:                       %d x %d x %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximum Grid dimensions:                        %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Warp size:                                      %d\n", prop.warpSize);
        printf("Maximum thread nums per block:                  %d\n", prop.warpSize);
        printf("Maximum Memory Bandwidth:                       %.f GB/s\n", prop.memoryClockRate*2.0*prop.memoryBusWidth/8/1e6);
        printf("Maximum Compute Ability:                        %.2f TFLOPs\n", prop.clockRate*prop.major*1000.0/1e9);
    }
    return 0;
}



