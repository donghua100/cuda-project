#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define blockThreadNum 512

void random_vec(float *A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = (rand() % 100)*0.1;
    }
}

void vec_add_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

int check(float *A, float *B, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(A[i] - B[i]) > 1e-5) {
            return -1;
        }
    }
    return 0;
}

// void __global__ vec_add(float *A, float *B, float *C, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         C[i] = A[i] + B[i];
//     }
// }

void __global__ vec_add(float *A, float *B, float *C, int n) {
    int stride = gridDim.x * blockDim.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = idx; i < n; i += stride) C[i] = A[i] + B[i];
}

int main() {
    srand(time(NULL));
    float *Ah, *Bh, *Ch, *Dh, *Eh, *Ad, *Bd, *Cd;
    int n = 100000000;

    Ah = (float *)malloc(sizeof(float)*n);
    Bh = (float *)malloc(sizeof(float)*n);
    Ch = (float *)malloc(sizeof(float)*n);
    Dh = (float *)malloc(sizeof(float)*n);
    Eh = (float *)malloc(sizeof(float)*n);
    random_vec(Ah, n);
    random_vec(Bh, n);
    clock_t start_h = clock();
    vec_add_cpu(Ah, Bh, Ch, n);
    clock_t end_h = clock();
    float t = (end_h - start_h)*1000.0/CLOCKS_PER_SEC;
    printf("CPU Time: %.2f ms, %.2f GFLOPs, %.2f GB/s\n", t, n/(t*1e6), 16*n/(t*1e6));

    cudaMalloc(&Ad, sizeof(float)*n);
    cudaMalloc(&Bd, sizeof(float)*n);
    cudaMalloc(&Cd, sizeof(float)*n);


    cudaMemcpy(Ad, Ah, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, Bh, sizeof(float)*n, cudaMemcpyHostToDevice);
    int gridBlockNum = (n - 1)/blockThreadNum + 1;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    vec_add<<<gridBlockNum, blockThreadNum>>>(Ad, Bd, Cd, n);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(Dh, Cd, sizeof(float)*n, cudaMemcpyDeviceToHost);

    if (check(Dh, Ch, n) < 0) {
        printf("Compute wrong answer\n");
    }
    float millisecond;
    cudaEventElapsedTime(&millisecond, start, end);
    printf("GPU Time: %.2f ms, %.2f GFLOPs, %.2f GB/s\n", millisecond, n/(millisecond*1e6), 16*n/(millisecond*1e6));


    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    cudaEventRecord(start);
    cublasSaxpy(handle, n, &alpha, Ad, 1, Bd, 1);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(Eh, Bd, sizeof(float)*n, cudaMemcpyDeviceToHost);
    if (check(Eh, Ch, n) < 0) {
        printf("Compute wrong answer\n");
    }
    cudaEventElapsedTime(&millisecond, start, end);
    printf("GPU Time: %.2f ms, %.2f GFLOPs, %.2f GB/s\n", millisecond, n/(millisecond*1e6), 16*n/(millisecond*1e6));

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);
    free(Ah);
    free(Bh);
    free(Ch);
    free(Dh);
    free(Eh);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return 0;
}
