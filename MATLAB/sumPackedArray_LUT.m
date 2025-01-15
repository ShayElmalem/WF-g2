function sumArray = sumPackedArray_LUT(packedArray)
% Function to sum the binary array along the third dimension for packed matrices using LUT

if isa(packedArray, 'gpuArray')
    temp = countOnesWithLUT_GPU(packedArray);
else
    temp = countOnesWithLUT(packedArray);
end
sumArray =  sum(temp,3);

end


