#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_MASK_WIDTH 125
__constant__ float M[MAX_MASK_WIDTH];

template <typename  T, size_t Width, size_t MaskWidth>
__global__ void conv_1d_basic(T *N, T *M, T *P) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0;
    int start = idx - MaskWidth/2;
    for (int i = 0; i < MaskWidth; i++) {
        if (start + i >= 0 && start + i < Width) {
            val += N[start + i]*M[i];
        }
    }
    // printf("%d\n", idx);
    if (idx < Width) P[idx] = val;
    // P[idx] = idx;

}

template<typename T, size_t Width, size_t MaskWidth>
__global__ void conv_1d_const(T *N, T *P) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0;
    int start = idx - MaskWidth/2;
    for (int i = 0; i < MaskWidth; i++) {
        if (start + i >= 0 && start + i < Width) {
            val += N[start + i]*M[i];
        }
    }
    if (idx < Width) P[idx] = val;
}


template<typename T, size_t Width, size_t MaskWidth, size_t TITLE_SIZE>
__global__ void conv_1d_share(T *N, T *P) {
    int left = blockDim.x*(blockIdx.x - 1) + threadIdx.x;
    int right = blockDim.x * (blockIdx.x + 1) + threadIdx.x;
    int tid = threadIdx.x;
    int r = MaskWidth/2;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float Nds[TITLE_SIZE + MaskWidth - 1];
    if (tid >= blockDim.x - r) Nds[tid - (blockDim.x - r)] = (left < 0)? 0:  N[left];
    Nds[r + tid] = N[idx];
    if (tid < r) Nds[r + blockDim.x + tid] = (right < Width)? N[right]: 0;
    __syncthreads();
    float val = 0;
    for (int i = 0; i < MaskWidth; i++) {
        val += Nds[tid + i]*M[i];
    }
    if (idx < Width) P[idx] = val;
}

template<typename T, size_t Width, size_t MaskWidth, size_t TITLE_SIZE>
__global__ void conv_1d_cache(T *N, T *P) {
    int title_start = blockDim.x * blockIdx.x;
    int nxt_title_start = blockDim.x * (blockIdx.x + 1);
    int tid = threadIdx.x;
    int r = MaskWidth/2;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float Nds[TITLE_SIZE];
    Nds[tid] = N[idx];
    __syncthreads();
    float val = 0;
    int start = idx - r;
    for (int i = 0; i < MaskWidth; i++) {
        int Nidx = start + i;
        if (Nidx >= 0 && Nidx < Width) {
            if (Nidx >= title_start && Nidx < nxt_title_start) {
                val += M[i]*Nds[tid + i - r];
            }
            else val += M[i]*N[Nidx];
        }
    }
    if (idx < Width) P[idx] = val;

}

template<typename T, size_t Width, size_t MaskWidth, size_t TITLE_SIZE>
__global__ void conv_1d_share_v2(T *N, T *P) {
    int idx = TITLE_SIZE * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int start = idx - MaskWidth/2;
    __shared__ float Nds[TITLE_SIZE + MaskWidth - 1];
    if (start>= 0 && start < Width) Nds[tid] = N[start];
    else Nds[tid] = 0;
    __syncthreads();
    float val = 0;
    if (tid < TITLE_SIZE) {
        for (int i = 0; i < MaskWidth; i++) {
            val += Nds[tid + i]*M[i];
        }
    }
    if (idx < Width) P[idx] = val;
}



#define CHECK(status) \
do { \
    cudnnStatus_t cuda_status = (status); \
    if (cuda_status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error encountered at line %d: %s\n", __LINE__, cudnnGetErrorString(cuda_status)); \
    } \
} while(0)

template <typename T, size_t Width, size_t MaskWidth>
void cudnn_conv_1d(T *N, T *M, T *P) {
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);

    cudnnDataType_t datatype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int input_dims[] = {1, 1, 1, Width };
    int output_dims[] = {1, 1, 1, Width };
    int kernel_dims[] = { 1, 1, 1, MaskWidth };
    int input_stride[] = {Width, Width, Width, 1};
    int output_stride[] = {Width, Width, Width, 1};

    // CHECK(cudnnSetTensor4dDescriptor(inputDesc, format, datatype, 1, 1, 1, Width));
    // CHECK(cudnnSetTensor4dDescriptor(outputDesc, format, datatype, 1, 1, 1, Width));
    // CHECK(cudnnSetFilter4dDescriptor(kernelDesc,  datatype, format, 1, 1, 1, MaskWidth));
    CHECK(cudnnSetTensorNdDescriptor(inputDesc, datatype, 4, input_dims, input_stride));
    CHECK(cudnnSetTensorNdDescriptor(outputDesc, datatype, 4, output_dims, output_stride));
    CHECK(cudnnSetFilterNdDescriptor(kernelDesc, datatype, format, 4, kernel_dims));

    int padA[] = {0, MaskWidth/2};
    int filterStrideA[] = {1, 1 };
    int dilationA[] = { 1, 1 };
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    //  cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    // CHECK(cudnnSetConvolution2dDescriptor(convDesc, 0,  MaskWidth/2, 1, 1, 1, 1, mode, datatype));
    CHECK(cudnnSetConvolutionNdDescriptor(convDesc, 2, padA, filterStrideA, dilationA, mode, datatype));

    int n, c, h, w;
    // CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, kernelDesc, &n, &c, &h, &w));
    int dim[4];
    CHECK(cudnnGetConvolutionNdForwardOutputDim(convDesc, inputDesc, kernelDesc, 4, dim));
    n = dim[0],c = dim[1], h = dim[2], w = dim[3];
    printf("n = %d, c = %d, h = %d, w = %d\n", n, c, h, w);
    // cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    // cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspaceSize = 0;
    float *workspace = NULL;
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize));
    cudaMalloc((void **)&workspace, workspaceSize);
    const float alpha = 1.0f, beta = 0.0f;
    CHECK(cudnnConvolutionForward(cudnnHandle, &alpha, inputDesc, N, kernelDesc, M, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, P));
    printf("workspaceSize = %lu\n", workspaceSize);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(kernelDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnnHandle);
    cudaFree(workspace);
}

template<typename T, size_t n>
void compare(T *a, T *b) {
    float err = 0;
    for (int i = 0; i < n; i++) {
        if (b[i] != 0) err += abs((a[i] - b[i])/b[i]);
    }
    err /= n;
    printf("The average error is %e\n", err);
}

template<size_t n>
void randarr(float *a) {
    for (int i = 0; i < n; i++) {
        a[i] = (rand() % 10) +  (rand() % 100) * 0.01;
    }
}

template<size_t n>
void printarr(float *a) {
    for (int i = 0; i < n; i++) {
        printf("%0.2f ", a[i]);
    }
    printf("\n");
}

int main()
{
    #define Width 102400
    #define MaskWidth MAX_MASK_WIDTH
    #define blockThreadNum 512
    float *Nh, *Mh, *Ph, *Phnn;
    Nh = (float *)malloc(sizeof(float) * Width);
    Mh = (float *)malloc(sizeof(float) * MaskWidth);
    Ph = (float *)malloc(sizeof(float) * Width);
    Phnn = (float *)malloc(sizeof(float) * Width);

    randarr<Width>(Nh);
    randarr<MaskWidth>(Mh);
    float *Nd, *Md, *Pd, *Pdnn;
    cudaMalloc((void **)&Nd, sizeof(float)*Width);
    cudaMalloc((void **)&Md, sizeof(float)*MaskWidth);
    cudaMalloc((void **)&Pd, sizeof(float)*Width);
    cudaMalloc((void **)&Pdnn, sizeof(float)*Width);
    cudaMemcpy(Nd, Nh, sizeof(float)*Width, cudaMemcpyHostToDevice);
    cudaMemcpy(Md, Mh, sizeof(float)*MaskWidth, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, Mh, MaskWidth*sizeof(float));
    int gridBlockNum = (Width - 1)/blockThreadNum + 1;
    conv_1d_share<float, Width, MaskWidth, blockThreadNum><<<gridBlockNum, blockThreadNum>>>(Nd, Pd);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudnn_conv_1d<float, Width, MaskWidth>(Nd, Md, Pdnn);
    cudaEventRecord(start);
    conv_1d_basic<float, Width, MaskWidth><<<gridBlockNum, blockThreadNum>>>(Nd, Md, Pd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millseconds = 0;
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  Width*MaskWidth*(1e-6)/millseconds);

    cudaEventRecord(start);
    conv_1d_const<float, Width, MaskWidth><<<gridBlockNum, blockThreadNum>>>(Nd, Pd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  Width*MaskWidth*(1e-6)/millseconds);

    cudaEventRecord(start);
    conv_1d_share<float, Width, MaskWidth, blockThreadNum><<<gridBlockNum, blockThreadNum>>>(Nd, Pd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  Width*MaskWidth*(1e-6)/millseconds);

    cudaEventRecord(start);
    conv_1d_cache<float, Width, MaskWidth, blockThreadNum><<<gridBlockNum, blockThreadNum>>>(Nd, Pd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  Width*MaskWidth*(1e-6)/millseconds);

    cudaEventRecord(start);
    conv_1d_share_v2<float, Width, MaskWidth, blockThreadNum><<< (Width - 1)/(blockThreadNum + MaskWidth - 1) + 1,blockThreadNum + MaskWidth - 1>>>(Nd, Pd);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  Width*MaskWidth*(1e-6)/millseconds);

    cudnn_conv_1d<float, Width, MaskWidth>(Nd, Md, Pdnn);
    // cudaEventRecord(start);
    // cudnn_conv_1d<float, Width, MaskWidth>(Nd, Md, Pdnn);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&millseconds, start, stop);
    // printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  Width*MaskWidth*(1e-6)/millseconds);

    cudaMemcpy(Ph, Pd, sizeof(float)*Width, cudaMemcpyDeviceToHost);
    cudaMemcpy(Phnn, Pdnn, sizeof(float)*Width, cudaMemcpyDeviceToHost);
    compare<float, Width>(Ph, Phnn);
    printarr<32>(Ph);
    printarr<32>(Phnn);
    // printarr<32>(Ph + Width - 1 - 32);
    // printarr<32>(Phnn + Width - 1 - 32);

    free(Nh);
    free(Mh);
    free(Ph);
    free(Phnn);

    cudaFree(Nd);
    cudaFree(Md);
    cudaFree(Pd);
    cudaFree(Pdnn);
}
