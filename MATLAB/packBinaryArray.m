function packedArray = packBinaryArray(binaryArray)
    % Function to pack binary array into uint8 array

    % Ensure the input array is of type uint8
    if ~isa(binaryArray, 'uint8')
        binaryArray = uint8(binaryArray);
    end

    % Validate that input is binary
    if ~(max(binaryArray(:)) <= 1 && min(binaryArray(:)) >= 0)
        error('Input array must contain only binary values (0 or 1).');
    end
    
    % Get size of the input array
    [M, N, K] = size(binaryArray);
    
    % Check if K is a multiple of 8, if not pad with zeros
    if mod(K, 8) ~= 0
        padSize = 8 - mod(K, 8);
        binaryArray = cat(3, binaryArray, zeros(M, N, padSize));
    end
    
    % Update K after padding
    K = size(binaryArray, 3);
    
    % Initialize the packed array
    packedArray = zeros(M, N, K/8, 'uint8');
    
    % Pack the binary array into uint8 array
    for i = 1:8:K
        for j = 0:7
            packedArray(:, :, (i+7)/8) = bitor(packedArray(:, :, (i+7)/8), ...
                bitshift(uint8(binaryArray(:, :, i+j)), j));
        end
    end
end

