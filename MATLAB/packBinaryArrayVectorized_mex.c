#include "mex.h"
#include <stdint.h>

void packBinaryArrayVectorized_mex(uint8_t* binaryArray, mwSize M, mwSize N, mwSize K, uint8_t* packedArray, bool dbgFlag) {
    mwSize paddedK = (K % 8 == 0) ? K : (K + (8 - K % 8));
    mwSize packedK = paddedK / 8;
    
    for (mwSize i = 0; i < M; i++) {
        for (mwSize j = 0; j < N; j++) {
            for (mwSize k = 0; k < packedK; k++) {
                uint8_t packedByte = 0;
                
                // Manually unrolling the loop to avoid overhead
                mwSize baseIdx = i + j * M + k * 8 * M * N;
                if (baseIdx < K) packedByte |= (binaryArray[baseIdx] << 0);
                if (baseIdx + M * N < K) packedByte |= (binaryArray[baseIdx + M * N] << 1);
                if (baseIdx + 2 * M * N < K) packedByte |= (binaryArray[baseIdx + 2 * M * N] << 2);
                if (baseIdx + 3 * M * N < K) packedByte |= (binaryArray[baseIdx + 3 * M * N] << 3);
                if (baseIdx + 4 * M * N < K) packedByte |= (binaryArray[baseIdx + 4 * M * N] << 4);
                if (baseIdx + 5 * M * N < K) packedByte |= (binaryArray[baseIdx + 5 * M * N] << 5);
                if (baseIdx + 6 * M * N < K) packedByte |= (binaryArray[baseIdx + 6 * M * N] << 6);
                if (baseIdx + 7 * M * N < K) packedByte |= (binaryArray[baseIdx + 7 * M * N] << 7);

                packedArray[i + j * M + k * M * N] = packedByte;

                if (dbgFlag) {
                    mexPrintf("Packed byte [%lu, %lu, %lu]: %02X\n", i, j, k, packedByte);
                }
            }
        }
    }
}

/* The gateway function */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("packBinaryArrayVectorized_mex:input", "At least one input required.");
    }

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("packBinaryArrayVectorized_mex:output", "One output required.");
    }

    uint8_t* binaryArray = (uint8_t*)mxGetData(prhs[0]);
    const mwSize* inputDims = mxGetDimensions(prhs[0]);
    mwSize M = inputDims[0];
    mwSize N = inputDims[1];
    mwSize K = (mxGetNumberOfDimensions(prhs[0]) > 2) ? inputDims[2] : 1;

    bool dbgFlag = false;
    if (nrhs > 1 && mxIsLogicalScalarTrue(prhs[1])) {
        dbgFlag = true;
    }

    mwSize paddedK = (K % 8 == 0) ? K : (K + (8 - K % 8));
    mwSize packedK = paddedK / 8;
    mwSize dims[3] = { M, N, packedK };

    plhs[0] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
    uint8_t* packedArray = (uint8_t*)mxGetData(plhs[0]);

    packBinaryArrayVectorized_mex(binaryArray, M, N, K, packedArray, dbgFlag);
}
