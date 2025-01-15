function sumArray = sumPackedArray(packedArray)
    % Function to sum the binary array along the third dimension for packed matrices using bit-level operations

    % Get size of the packed array
    [M, N, ~] = size(packedArray);
    
    % Initialize the sum array
    sumArray = zeros(M, N);
    
    % Loop through each bit position
    for bitPos = 0:7
        % Extract the bit at the current position and add it to the sum
        sumArray = sumArray + sum(bitget(packedArray, bitPos + 1), 3);
    end

end


