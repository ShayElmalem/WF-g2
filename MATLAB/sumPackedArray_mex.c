#include "mex.h"
#include <stdint.h>

// Lookup table for the number of '1' bits in each uint8_t value
static const uint8_t bitCountLookup[256] = {
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

void sumPackedArray_mex(double *sumArray, const uint8_t *packedArray, mwSize M, mwSize N, mwSize P, bool dbgFlag) {
    mwSize i, j, k;
    
    // Initialize sumArray to zero
    for (i = 0; i < M * N; i++) {
        sumArray[i] = 0.0;
    }
    
    // Sum the bits in packedArray along the third dimension
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < P; k++) {
                uint8_t val = packedArray[i + j * M + k * M * N];
                sumArray[i + j * M] += bitCountLookup[val];
                
                // Debug prints if dbgFlag is set
                if (dbgFlag) {
                    mexPrintf("Processing element (%lu, %lu, %lu) with value %u, sum: %f\n", 
                              i + 1, j + 1, k + 1, val, sumArray[i + j * M]);
                }
            }
        }
    }
}

// MATLAB entry point
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Validate input arguments
    if (nrhs < 1 || nrhs > 2) {
        mexErrMsgIdAndTxt("sumPackedArray_mex:invalidNumInputs", "One or two inputs required.");
    }
    
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("sumPackedArray_mex:invalidNumOutputs", "One output required.");
    }
    
    // Get the input packed array
    const mxArray *packedArrayMx = prhs[0];
    const uint8_t *packedArray = (const uint8_t *)mxGetData(packedArrayMx);
    
    // Get dimensions of the input array
    mwSize M = mxGetDimensions(packedArrayMx)[0];
    mwSize N = mxGetDimensions(packedArrayMx)[1];
    mwSize P = mxGetDimensions(packedArrayMx)[2];
    
    // Get the debug flag if provided
    bool dbgFlag = false;
    if (nrhs == 2) {
        dbgFlag = mxIsLogicalScalarTrue(prhs[1]);
    }
    
    // Create the output array
    plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL);
    double *sumArray = mxGetPr(plhs[0]);
    
    // Call the sum function
    sumPackedArray_mex(sumArray, packedArray, M, N, P, dbgFlag);
}
