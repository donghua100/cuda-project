#include <cuda_runtime.h>
#include <stdio.h>

#define blockThreadNum 512
template<typename T, size_t N>
__global__ void reduce_basic(T *in, T *out) {
    __shared__ T sdata[blockThreadNum];
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) sdata[tx] = in[idx];
    else sdata[tx] = 0;
    __syncthreads();

    for (int s = 1; s < blockThreadNum; s *= 2) {
        if (tx % (2*s) == 0) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0) out[blockIdx.x] = sdata[0];

}

template<typename T, size_t N>
__global__ void reduce1(T *in, T *out) {
    __shared__ T sdata[blockThreadNum];
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) sdata[tx] = in[idx];
    else sdata[tx] = 0;
    __syncthreads();

    for (int s = 1; s < blockThreadNum; s *= 2) {
        idx = 2*s*tx;
        if (idx < blockThreadNum) {
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }
    if (tx == 0) out[blockIdx.x] = sdata[0];
}


template<typename T, size_t N>
__global__ void reduce2(T *in, T *out) {
    __shared__ T sdata[blockThreadNum];
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) sdata[tx] = in[idx];
    else sdata[tx] = 0;
    __syncthreads();

    for (int s = blockThreadNum / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0) out[blockIdx.x] = sdata[0];
}

template<typename T, size_t N, size_t blockThreadNums>
__global__ void reduce3(T *in, T *out) {
    __shared__ T sdata[blockThreadNums];
    int tx = threadIdx.x;
    int i = blockIdx.x*(2*blockDim.x) + threadIdx.x;
    if (i < N) sdata[tx] = in[i];
    else sdata[tx] = 0;
    if (i + blockDim.x < N) sdata[tx] += in[i + blockDim.x];
    __syncthreads();
    for (int s = blockThreadNums/2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0) out[blockIdx.x] = sdata[0];
}

template<typename T, size_t n>
void rand_arr(T *a) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 10;
    }
}

template<typename T, size_t n>
void one_arr(T *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
}

template<typename T, size_t n>
void print_arr(T *a) {
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}
#define CUDA_CHECK(status) \
do { \
    cudaError_t cuda_status = (status); \
    if (cuda_status != cudaSuccess) { \
        printf("cuda error encountered at line %d: %s\n", __LINE__, cudaGetErrorString(cuda_status)); \
    } \
} while(0)

int main() {
    #define N 1024000
    typedef int T;
    #define gridblock  ((N - 1)/blockThreadNum + 1)
    T *h_in, *h_out;
    T *d_in, *d_out;
    h_in = (T *)malloc(sizeof(T)*N);
    h_out = (T *)malloc(sizeof(T)*gridblock);

    rand_arr<T, N>(h_in);
    one_arr<T, N>(h_in);
    T s = 0;
    for (int i = 0; i < N; i++) s += h_in[i];
    CUDA_CHECK(cudaMalloc((void **)&d_in, sizeof(T)*N));
    CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(T)*gridblock));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(T)*N, cudaMemcpyHostToDevice));

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce_basic<T, N><<<gridblock, blockThreadNum>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millseconds;
    cudaEventElapsedTime(&millseconds, start, stop);
    // printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  N*(1e-6)/millseconds);
    printf("Time %.2f ms,  %.2f GB/s\n",millseconds,  N*sizeof(T)*1000.0/1024/1024/1024/millseconds);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(T)*gridblock, cudaMemcpyDeviceToHost));


    cudaEventRecord(start);
    reduce1<T, N><<<gridblock, blockThreadNum>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GB/s\n",millseconds,  N*sizeof(T)*1000.0/1024/1024/1024/millseconds);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(T)*gridblock, cudaMemcpyDeviceToHost));

    cudaEventRecord(start);
    reduce2<T, N><<<gridblock, blockThreadNum>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GB/s\n",millseconds,  N*sizeof(T)*1000.0/1024/1024/1024/millseconds);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(T)*gridblock, cudaMemcpyDeviceToHost));

    cudaEventRecord(start);
    #define blockThreadNums (blockThreadNum/2)
    reduce3<T, N, blockThreadNums><<<gridblock, blockThreadNums>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GB/s\n",millseconds,  N*sizeof(T)*1000.0/1024/1024/1024/millseconds);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(T)*gridblock, cudaMemcpyDeviceToHost));

    T ss = 0;
    for (int i = 0; i < gridblock; i++) ss += h_out[i];
    printf("%d %d\n", s, ss);
    // print_arr<T, gridblock>(h_out);

    return 0;
}
