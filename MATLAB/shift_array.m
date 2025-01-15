function shifted_array = shift_array(array, shifts)
    % Perform circular shift using circshift
    shifted_array = circshift(array, shifts);
    
    % Loop over each dimension to zero out the unwanted values
    for dim = 1:length(shifts)
        shift = shifts(dim);
        sz = size(array, dim);
        
        if shift > 0
            % Right shift (positive shift) - zero out the beginning of the dimension
            idx = repmat({':'}, 1, ndims(array));
            idx{dim} = 1:shift;
            shifted_array(idx{:}) = 0;
        elseif shift < 0
            % Left shift (negative shift) - zero out the end of the dimension
            idx = repmat({':'}, 1, ndims(array));
            idx{dim} = (sz+shift+1):sz;
            shifted_array(idx{:}) = 0;
        end
    end
end
