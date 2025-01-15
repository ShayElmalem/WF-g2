#include "mex.h"
#include <stdint.h>  // Include this header for uint8_t

void countOnesWithLUT(uint8_t *inputArray, mwSize numElements, uint8_t *outputArray, uint8_t *LUT) {
    for (mwSize i = 0; i < numElements; i++) {
        outputArray[i] = LUT[inputArray[i]];
    }
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Check for proper number of arguments. */
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidNumInputs",
                          "One input required.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:maxlhs",
                          "Too many output arguments.");
    }

    /* Check if the input is a 3D uint8 array. */
    if (!mxIsUint8(prhs[0]) || mxGetNumberOfDimensions(prhs[0]) != 3) {
        mexErrMsgIdAndTxt("MATLAB:mexFunction:invalidInput",
                          "Input must be a 3D uint8 array.");
    }

    /* Get the input array. */
    uint8_t *inputArray = (uint8_t *)mxGetData(prhs[0]);
    
    /* Get the number of elements in the input array. */
    mwSize numElements = mxGetNumberOfElements(prhs[0]);

    /* Get the dimensions of the input array. */
    const mwSize *dims = mxGetDimensions(prhs[0]);

    /* Create the output array with the same dimensions as the input. */
    plhs[0] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
    uint8_t *outputArray = (uint8_t *)mxGetData(plhs[0]);

    /* Create the LUT for counting ones in binary representation. */
    uint8_t LUT[256];
    for (int i = 0; i < 256; i++) {
        uint8_t value = (uint8_t)i;
        uint8_t count = 0;
        while (value) {
            count += value & 1;
            value >>= 1;
        }
        LUT[i] = count;
    }

    /* Call the function to count the ones using LUT. */
    countOnesWithLUT(inputArray, numElements, outputArray, LUT);
}
