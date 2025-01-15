function packedArray = packBinaryArrayVectorized(binaryArray)
% Function to pack binary array into uint8 array using vectorization

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
    binaryArray = cat(3, binaryArray, zeros(M, N, padSize, 'uint8'));
end

% Update K after padding
K = size(binaryArray, 3);

% Initialize the packed array
packedArray = zeros(M, N, K/8, 'uint8');

% Create a bit shift matrix
bitShifts = uint8(reshape(2.^(0:7), 1, 1, 8));

% Reshape the binary array for multiplication
binaryArrayReshaped = reshape(binaryArray, M, N, 8, []);

% Perform the pointwise multiplication and sum over the third dimension
packedArray = squeeze(uint8(sum(binaryArrayReshaped .* bitShifts, 3)));
end
