/*
 * Author: Lee Namgoo
 * E-Mail: lee.namgoo@sualab.com
 */

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <time.h>
#include <math.h>
#include <iostream>

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
#define CLIP(a) (a < 0U ? 0U : a > 255U ? 255U : a)

/**
 * The __global__ qualifier declares a function as being a kernel.
 *   1. Executed on the device
 *   2. Callable from the host
 *   3. Must have void return type
 *   4. Must specify its execution configuration
 *   5. Call to this function is asynchronous, it returns before the device has
 *      completed its execution.
 */
__global__ void
SobelX(unsigned char* const input, unsigned char* const output, const int width,
        const int height, const int input_step, const int output_step)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_center = input_step * yIndex + xIndex;
    const int output_index = output_step * yIndex + xIndex;

    if ((xIndex == 0) || (xIndex == width - 1)
            || (yIndex == 0) || (yIndex == height - 1)) {

        output[output_index] = input[input_center];
        return;
    }

    if ((xIndex < width) && (yIndex < height)) {

        int tmp = input_center - input_step;
        int tmp2 = input_center + input_step;

        int sobel_x = input[tmp - 1] + input[tmp] * 2 + input[tmp + 1]
            - input[tmp2 - 1] - input[tmp2] * 2 - input[tmp2 + 1];

        output[output_index] = CLIP(sobel_x);
    }
}

__global__ void
SobelY(unsigned char* const input, unsigned char* const output, const int width,
        const int height, const int input_step, const int output_step)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_center = input_step * yIndex + xIndex;
    const int output_index = output_step * yIndex + xIndex;

    if ((xIndex == 0) || (xIndex == width - 1)
            || (yIndex == 0) || (yIndex == height - 1)) {

        output[output_index] = input[input_center];
        return;
    }

    if ((xIndex < width) && (yIndex < height)) {

        int tmp = input_center - 1;
        int tmp2 = input_center + 1;

        int sobel_y = input[tmp - input_step] + input[tmp] * 2 + input[tmp + input_step]
            - input[tmp2 - input_step] - input[tmp2] * 2 - input[tmp2 + input_step];

        output[output_index] = CLIP(sobel_y);
    }
}

__global__ void
Sobel(unsigned char* const input, unsigned char* const output, const int width,
        const int height, const int input_step, const int output_step)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int input_center = input_step * yIndex + xIndex;
    const int output_index = output_step * yIndex + xIndex;

    if ((xIndex == 0) || (xIndex == width - 1)
            || (yIndex == 0) || (yIndex == height - 1)) {

        output[output_index] = input[input_center];
        return;
    }

    const int SobelX[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    const int SobelY[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

    if ((xIndex < width) && (yIndex < height)) {

        /* Assembly Level Optimization Possible */
        int sumX = 0, sumY = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int data = input[input_center + i * input_step + j];
                sumX += data * SobelX[(i + 1) * 3 + j + 1];
                sumY += data * SobelY[(i + 1) * 3 + j + 1];
            }
        }
        output[output_index] = sqrt((double)(sumX * sumX + sumY * sumY) / 32);
    }
}
/**
 * The __host__ qualifier declares a function that is
 *   1. Executed on the host
 *   2. Callable from the host only
 */
__host__ bool
sobel_cuda(const cv::Mat& h_input, cv::Mat& h_output, int type)
{
    unsigned char* d_input;
    unsigned char* d_output;

    /* Note that input and output Mats' step may differ */
    const int input_size = h_input.step * h_input.rows;
    const int output_size = h_output.step * h_output.rows;

    SAFE_CALL(cudaMalloc<unsigned char>(&d_input, input_size), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output, output_size), "CUDA Malloc Failed");

    SAFE_CALL(cudaMemcpy(d_input, h_input.ptr(), input_size, cudaMemcpyHostToDevice),
            "CUDA Memcpy Host To Device Failed");

    const dim3 threadsPerBlock(32, 32); /* upper limit : 1024 */
    const dim3 blocksPerGrid(
            (h_output.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (h_output.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    clock_t begin = clock();

    switch (type) {
    case 0:
        cudaEventRecord(start);
        Sobel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, h_output.cols,
             h_output.rows, h_input.step, h_output.step);
        cudaEventRecord(stop);
        break;
    case 1:
        SobelX<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, h_output.cols,
             h_output.rows, h_input.step, h_output.step);
        break;
    case 2:
        SobelY<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, h_output.cols,
             h_output.rows, h_input.step, h_output.step);
        break;
    }

    /* Wait until CUDA is done */
    SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");
    SAFE_CALL(cudaMemcpy(h_output.ptr(), d_output, output_size, cudaMemcpyDeviceToHost),
            "CUDA Memcpy Device To Host Failed");

    std::cout << "Kernel execution time : " << (float)(clock() - begin) /
        (CLOCKS_PER_SEC / 1000) << "ms\n";

    //cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time (CUDA event) : " << milliseconds <<
        "ms\n";

    SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
    SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");

    return true;
}
