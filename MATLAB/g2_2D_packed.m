function [g0, G0, Ginf, gt] = g2_2D_packed(img, g2range, conn_map, chunkMode, chunkSz, gpuFlag)

%% genereal settings
if nargin < 4 || isempty(chunkMode)
    chunkMode = false;  % Set default value
end

if nargin < 5 || isempty(chunkSz)
    chunkSz = 1000;  % Set default value, corresponds to 1e5/100 in the origianl test
end

if nargin < 6 || isempty(gpuFlag)
    gpuFlag = false;  % Set default value
end

if iscell(conn_map)
    multiConnFlag = true;
    nConn = length(conn_map);
else
    multiConnFlag = false;
end
inpSz = size(img);

%% validate that all conn_maps are with single neighbor only (due to packing assumptions)

if multiConnFlag
    for k = 1:nConn
        chk = sum(conn_map{k}(:));
        if (chk ~= 1)
            error('conn_map have to include single neighbor for the g2_packed function');
        end
    end
else
    chk = sum(conn_map(:));
    if (chk ~= 1)
        error('conn_map have to include single neighbor for the g2_packed function');
    end
end

%%
if ~chunkMode
    if multiConnFlag
        spat_corr = cell(1,nConn);
        for k = 1:nConn
            [row,col] = find(conn_map{k});
            spat_corr{k} = shift_array(img, [row-2, col-2, 0]);
        end
    else
        spat_corr = imfilter(img,rot90(conn_map,2),0);
    end
end


%%

% prepare empty tensor
if multiConnFlag
    gt = zeros([inpSz(1:2) g2range+1 nConn],'uint32');
else
    gt = zeros([inpSz(1:2) g2range+1],'uint32');
end

if chunkMode
    nChunk = ceil(inpSz(3)/chunkSz); % number of chunks
    maxT = inpSz(3); % max of temporal dimmention

    if multiConnFlag
        conn_map = cat(3,conn_map{:});
    end
    
    if gpuFlag
        conn_map = gpuArray(conn_map); % connectiviy map to GPU
    end

    if gpuFlag
        curGt = gpuArray(zeros([inpSz(1:2) g2range+1, nConn],'like', gt));
    else
        curGt = zeros([inpSz(1:2) g2range+1, nConn],'like', gt);
    end

    for m = 1:nChunk
        curIdx = (1 + (m-1)*chunkSz) : min([chunkSz*m,maxT]);
        if gpuFlag
            curImg = gpuArray(img(:,:,curIdx));
        else
            curImg = img(:,:,curIdx);
        end

        if multiConnFlag
            %packedCurSpatCorr = zeros([inpSz(1) inpSz(2) ceil(length(curIdx)/8) nConn],'like',curImg);
            packedCurSpatCorr = cell(1,nConn);
            for k = 1:nConn
                [row,col] = find(conn_map(:,:,k));
                temp = shift_array(curImg, [row-2, col-2, 0]);
                
                if gpuFlag
                    packedCurSpatCorr{k} = packBinaryArray_GPU(temp);
                else
                    packedCurSpatCorr{k} = packBinaryArrayVectorized(temp);
                end
            end
            clear temp
        else
            curSpatCorr = imfilter(curImg,rot90(conn_map,2),0);
            if gpuFlag
                packedCurSpatCorr = packBinaryArray_GPU(curSpatCorr);
            else
                packedCurSpatCorr = packBinaryArrayVectorized(curSpatCorr);
            end
            clear curSpatCorr
        end

        % pack 0:7 shifts for packed shift
        packedArrayShift = cell(1,8);
        if gpuFlag
            packedArrayShift{1} = packBinaryArray_GPU(curImg);
        else
            packedArrayShift{1} = packBinaryArrayVectorized(curImg);
        end
        for k = 2:8
            temp_shft = circshift(curImg,[0,0,(k-1)]);
            if gpuFlag
                packedArrayShift{k} = packBinaryArray_GPU(temp_shft);
            else
                packedArrayShift{k} = packBinaryArrayVectorized(temp_shft);
            end
        end
        clear curImg temp_shft

        curGt(:) = 0;
        for k = 0:g2range % loop over delays

            % img tensor at delay k
            % replacing 't = circshift(curSpatCorr,[0,0,-k]);', but shifting img
            % and not spt_corr
            shftVal_int = floor(k/8);
            shftVal_rem = rem(k,8);
            packedShiftRes = circshift(packedArrayShift{shftVal_rem+1},[0,0,shftVal_int]);

            if multiConnFlag
                for n = 1:nConn
                    % correlate with original tensor
                    % replacing 'gt(:,:,k+1) = sum(img.*t, 3,"omitnan");'
                    chk = packedCurSpatCorr{n};
                    tempPackedMultResult = bitand(chk, packedShiftRes);

                    %curGt(:,:,k+1,n) = sumPackedArray(tempPackedMultResult);
                    curGt(:,:,k+1,n) = sumPackedArray_LUT(tempPackedMultResult);
                end
            else % ~multiConn
                % correlate with original tensor
                % replacing 'curGt(:,:,k+1) = sum(curImg.*t, 3,"omitnan");'
                tempPackedMultResult = bitand(packedCurSpatCorr, packedShiftRes);
                    
                %curGt(:,:,k+1) = sumPackedArray(tempPackedMultResult);    
                curGt(:,:,k+1) = sumPackedArray_LUT(tempPackedMultResult);

            end
        end
        if gpuFlag
            gt = gt + gather(curGt);
        else
            gt = gt + curGt;
        end
    end

else % sequential

    % packing spat_corr
    if multiConnFlag
        packedSpatCorr = cell(1,nConn);
        for k = 1:nConn
            packedSpatCorr{k} = packBinaryArrayVectorized(spat_corr{k});
        end
    else
        packedSpatCorr = packBinaryArrayVectorized(spat_corr);
    end

    % pack 0:7 shifts for packed shift
    packedArrayShift{1} = packBinaryArrayVectorized(img);
    for k = 8:-1:2
        temp_shft = circshift(img,[0,0,(k-1)]);
        packedArrayShift{k} = packBinaryArrayVectorized(temp_shft);
    end

    for k = 0:g2range % loop over delays

        % img tensor at delay k
        % replacing 't = circshift(spat_corr,[0,0,-k]);', but shifting img
        % and not spt_corr
        shftVal_int = floor(k/8);
        shftVal_rem = rem(k,8);
        packedShiftRes = circshift(packedArrayShift{shftVal_rem+1},[0,0,shftVal_int]);

        if multiConnFlag
            for m = 1:nConn
                % correlate with original tensor
                % replacing 'gt(:,:,k+1) = sum(img.*t, 3,"omitnan");'
                tempPackedMultResult = bitand(packedSpatCorr{m}, packedShiftRes);
                %gt(:,:,k+1,m) = sumPackedArray(tempPackedMultResult);
                gt(:,:,k+1,m) = sumPackedArray_LUT(tempPackedMultResult);
            end

        else
            % correlate with original tensor
            % replacing 'gt(:,:,k+1) = sum(img.*t, 3,"omitnan");'
            tempPackedMultResult = bitand(packedSpatCorr, packedShiftRes);
            
            %gt(:,:,k+1) = sumPackedArray(tempPackedMultResult);
            gt(:,:,k+1) =  sumPackedArray_LUT(tempPackedMultResult);
            
        end
    end

end


gt = double(gt);


%%
if multiConnFlag
    % XX
    G0 = [];
    Ginf = [];
    g0 = [];
else
    G0 = squeeze(gt(:,:,1));
    Ginf = squeeze(mean(gt(:,:,2:end),3));

    %normalized
    g0 = G0./Ginf;

end