function binaryArray = unpackBinaryArray(packedArray, originalK)
    % Function to unpack uint8 array into binary array

    % Get size of the packed array
    [M, N, packedK] = size(packedArray);
    
    % Initialize the binary array
    binaryArray = false(M, N, packedK*8);
    
    % Unpack the uint8 array into binary array
    for i = 1:packedK
        for j = 0:7
            binaryArray(:, :, (i-1)*8 + j + 1) = bitget(packedArray(:, :, i), j + 1);
        end
    end
    
    % Remove padding if necessary
    if nargin > 1 && originalK < size(binaryArray, 3)
        binaryArray = binaryArray(:, :, 1:originalK);
    end
end
