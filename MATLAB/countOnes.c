#include "mex.h"
#include <stdint.h>  // Include this header for uint8_t

void countOnes(uint8_t *inputArray, mwSize numElements, uint8_t *outputArray) {
    for (mwSize i = 0; i < numElements; i++) {
        uint8_t value = inputArray[i];
        uint8_t count = 0;
        while (value) {
            count += value & 1;
            value >>= 1;
        }
        outputArray[i] = count;
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

    /* Get the input array. */
    uint8_t *inputArray = (uint8_t *)mxGetData(prhs[0]);
    
    /* Get the number of elements in the input array. */
    mwSize numElements = mxGetNumberOfElements(prhs[0]);

    /* Create the output array. */
    plhs[0] = mxCreateNumericMatrix(1, numElements, mxUINT8_CLASS, mxREAL);
    uint8_t *outputArray = (uint8_t *)mxGetData(plhs[0]);

    /* Call the function to count the ones. */
    countOnes(inputArray, numElements, outputArray);
}
