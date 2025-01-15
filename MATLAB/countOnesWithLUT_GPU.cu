#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>

// Define the lookup table for counting the number of ones in a byte
__device__ __constant__ unsigned char lookupTable[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

// Kernel function to process each element using the lookup table
__global__ void countOnesKernel(const unsigned char* input, unsigned char* output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = lookupTable[input[idx]];
    }
}

// Entry point for the MEX function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Initialize the GPU API
    mxInitGPU();

    // Get the input gpuArray
    mxGPUArray const *inputGPUArray = mxGPUCreateFromMxArray(prhs[0]);
    const unsigned char *input = (const unsigned char *)(mxGPUGetDataReadOnly(inputGPUArray));

    // Get the dimensions of the input array
    mwSize const *dims = mxGPUGetDimensions(inputGPUArray);
    size_t N = mxGPUGetNumberOfElements(inputGPUArray);

    // Create the output gpuArray
    mxGPUArray *outputGPUArray = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(inputGPUArray), 
                                                     dims, 
                                                     mxUINT8_CLASS, 
                                                     mxREAL, 
                                                     MX_GPU_DO_NOT_INITIALIZE);
    unsigned char *output = (unsigned char *)(mxGPUGetData(outputGPUArray));

    // Set up the execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    countOnesKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

    // Wrap the output gpuArray into an mxArray and return it
    plhs[0] = mxGPUCreateMxArrayOnGPU(outputGPUArray);

    // Destroy the input and output gpuArray
    mxGPUDestroyGPUArray(inputGPUArray);
    mxGPUDestroyGPUArray(outputGPUArray);
}
