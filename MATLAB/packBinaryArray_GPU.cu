#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <stdint.h> // Include this header for uint8_t

#define BITS_PER_BYTE 8

// CUDA kernel to pack binary values into uint8
__global__ void packBinaryKernel(const uint8_t *input, uint8_t *output, size_t M, size_t N, size_t K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < M && y < N && z < (K / BITS_PER_BYTE)) {
        uint8_t packedValue = 0;

        for (int bit = 0; bit < BITS_PER_BYTE; ++bit) {
            int originalZ = z * BITS_PER_BYTE + bit;
            uint8_t bitValue = input[x + y * M + originalZ * M * N];
            packedValue |= (bitValue << bit);
        }

        output[x + y * M + z * M * N] = packedValue;
    }
}

// MEX gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Initialize the GPU API
    mxInitGPU();

    // Check for proper number of arguments
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("packBinaryArray_GPU:invalidNumInputs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("packBinaryArray_GPU:invalidNumOutputs", "One output required.");
    }

    // Get input 3D array
    const mxGPUArray *input = mxGPUCreateFromMxArray(prhs[0]);

    // Ensure the input is a uint8 array
    if (mxGPUGetClassID(input) != mxUINT8_CLASS) {
        mexErrMsgIdAndTxt("packBinaryArray_GPU:invalidInput", "Input must be a uint8 array.");
    }

    const mwSize *dims = mxGPUGetDimensions(input);
    size_t M = dims[0];
    size_t N = dims[1];
    size_t K = dims[2];

    if (K % BITS_PER_BYTE != 0) {
        mexErrMsgIdAndTxt("packBinaryArray_GPU:invalidDimension", "The third dimension (K) must be divisible by 8.");
    }

    // Create output 3D array
    mwSize outputDims[3] = {M, N, K / BITS_PER_BYTE};
    mxGPUArray *output = mxGPUCreateGPUArray(3, outputDims, mxUINT8_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    const uint8_t *inputData = (const uint8_t *)mxGPUGetDataReadOnly(input);
    uint8_t *outputData = (uint8_t *)mxGPUGetData(output);

    // Define grid and block sizes
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (K / BITS_PER_BYTE + blockDim.z - 1) / blockDim.z);

    // Launch the CUDA kernel
    packBinaryKernel<<<gridDim, blockDim>>>(inputData, outputData, M, N, K);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Return the output
    plhs[0] = mxGPUCreateMxArrayOnGPU(output);

    // Destroy GPU arrays
    mxGPUDestroyGPUArray(input);
    mxGPUDestroyGPUArray(output);
}
