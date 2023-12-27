#include <cuda_runtime.h>
#include<cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_MASK_WIDTH 5
__constant__ float M[MAX_MASK_WIDTH][MAX_MASK_WIDTH];

template<typename T, size_t height, size_t width, size_t MaskWidth, size_t TITLE_WIDTH>
__global__ void conv2d(T *N, T *P, int ld) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = TITLE_WIDTH * blockIdx.y + ty;
    int col_o = TITLE_WIDTH * blockIdx.x + tx;
    int row_i = row_o;
    int col_i = col_o;
    __shared__ T Nds[TITLE_WIDTH + MaskWidth - 1][TITLE_WIDTH + MaskWidth - 1];
    if (row_i < height && col_i < width) {
        Nds[ty][tx] = N[row_i * ld + col_i];
    }
    __syncthreads();
    // if (tx < 5 && ty < 5)
    // if (tx < TITLE_WIDTH && ty < TITLE_WIDTH && row_o < height && col_o < width) {
    //     printf("blockIdxY = %d, blockIdxX = %d, threadIdxY = %d, threadIdxX = %d\n", blockIdx.y, blockIdx.x, ty, tx);
    //     printf("row_o = %d, col_o = %d\n", row_o, col_o);
    //     P[row_o * ld + col_o] = 1;
    // }
    T val = 0;
    if (tx < TITLE_WIDTH && ty < TITLE_WIDTH) {
        for (int i = 0; i < MaskWidth; i++) {
            for (int j = 0; j < MaskWidth; j++) {
                val += M[i][j] * Nds[i + ty][j + tx];
            }
        }
        if (row_o < height - MaskWidth + 1 && col_o < width - MaskWidth + 1) {
            P[row_o*ld + col_o] = val;
        }
    }
}


#define CHECK(status) \
do { \
    cudnnStatus_t cuda_status = (status); \
    if (cuda_status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error encountered at line %d: %s\n", __LINE__, cudnnGetErrorString(cuda_status)); \
    } \
} while(0)


#define CUDA_CHECK(status) \
do { \
    cudaError_t cuda_status = (status); \
    if (cuda_status != cudaSuccess) { \
        printf("cuda error encountered at line %d: %s\n", __LINE__, cudaGetErrorString(cuda_status)); \
    } \
} while(0)


template <typename T, size_t height, size_t width, size_t MaskWidth>
void cudnn_conv_2d(T *N, T *M, T *P, int ld) {
    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&kernelDesc);

    cudnnDataType_t datatype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int input_dims[] = {1, 1, height, width };
    int output_dims[] = {1, 1, height - MaskWidth + 1, width - MaskWidth + 1};
    int kernel_dims[] = { 1, 1, MaskWidth, MaskWidth };
    int input_stride[] = {height *ld, height*ld, ld, 1};
    int output_stride[] = {(height - MaskWidth + 1)*ld, (height - MaskWidth + 1)*ld, ld, 1};

    // CHECK(cudnnSetTensor4dDescriptor(inputDesc, format, datatype, 1, 1, 1, Width));
    // CHECK(cudnnSetTensor4dDescriptor(outputDesc, format, datatype, 1, 1, 1, Width));
    // CHECK(cudnnSetFilter4dDescriptor(kernelDesc,  datatype, format, 1, 1, 1, MaskWidth));
    CHECK(cudnnSetTensorNdDescriptor(inputDesc, datatype, 4, input_dims, input_stride));
    CHECK(cudnnSetTensorNdDescriptor(outputDesc, datatype, 4, output_dims, output_stride));
    CHECK(cudnnSetFilterNdDescriptor(kernelDesc, datatype, format, 4, kernel_dims));

    int padA[] = {0, 0};
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
    #define height 1000
    #define width 1000
    #define num height * width
    #define MaskWidth MAX_MASK_WIDTH
    #define blockThreadNum (32 - MaskWidth)
    #define outheight (height - MaskWidth + 1)
    #define outwidth (width - MaskWidth + 1)
    #define outnum (outheight)*(outwidth)
    float *Nh, *Mh, *Ph, *Phnn;
    Nh = (float *)malloc(sizeof(float) * num);
    Mh = (float *)malloc(sizeof(float) * MaskWidth*MaskWidth);
    Ph = (float *)malloc(sizeof(float) * outnum);
    Phnn = (float *)malloc(sizeof(float) * outnum);

    randarr<num>(Nh);
    randarr<MaskWidth * MaskWidth>(Mh);
    printarr<MaskWidth*MaskWidth>(Mh);
    float *Nd, *Md, *Pd, *Pdnn;
    size_t pitch, pitch_dnn;
    // pitch = width;
    CUDA_CHECK(cudaMallocPitch((void **)&Nd, &pitch, sizeof(float)*(width), height));
    printf("pitch = %lu\n", pitch);
    // cudaMalloc((void **)&Nd, sizeof(float)*num);
    CUDA_CHECK(cudaMalloc((void **)&Md, sizeof(float)*MaskWidth * MaskWidth));
    CUDA_CHECK(cudaMallocPitch((void **)&Pd, &pitch, sizeof(float)*outwidth, outheight));
    // cudaMalloc((void **)&Pd, sizeof(float)*width*height);
    CUDA_CHECK(cudaMallocPitch((void **)&Pdnn, &pitch_dnn, sizeof(float)*outwidth, outheight));
    CUDA_CHECK(cudaMemcpy2D(Nd, pitch, Nh, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
    printf("pitch = %lu\n", pitch);
    // cudaMemcpy(Nd, Nh, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaMemcpy(Md, Mh, sizeof(float)*MaskWidth * MaskWidth, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(M, Mh, MaskWidth * MaskWidth*sizeof(float)));
    int blockdimx = blockThreadNum + MaskWidth - 1;
    int blockdimy = blockThreadNum + MaskWidth - 1;
    dim3 blockThread = dim3(blockdimx, blockdimy);
    dim3 gridBlock = dim3((width-1)/blockThreadNum + 1,(height - 1)/blockThreadNum + 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2d<float, height, width, MaskWidth, blockThreadNum><<<gridBlock, blockThread>>>(Nd, Pd, pitch/sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millseconds = 0;
    cudaEventElapsedTime(&millseconds, start, stop);
    printf("Time %.2f ms,  %.2f GFLOPS\n",millseconds,  num*MaskWidth*MaskWidth*(1e-6)/millseconds);

    cudnn_conv_2d<float, height, width, MaskWidth>(Nd, Md, Pdnn, pitch_dnn/sizeof(float));
    CUDA_CHECK(cudaMemcpy2D(Ph, sizeof(float)*outwidth, Pd, pitch, sizeof(float)*outwidth, outheight, cudaMemcpyDeviceToHost));
    // cudaMemcpy(Ph, Pd, sizeof(float)*num, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(Phnn, sizeof(float)*outwidth, Pdnn, pitch_dnn, sizeof(float)*outwidth, outheight, cudaMemcpyDeviceToHost);
    compare<float, num>(Ph, Phnn);
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
