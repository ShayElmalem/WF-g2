function [g0, G0, Ginf, gt] = g2_2D(img, g2range, conn_map, chunkMode, chunkSz, gpuFlag)

if nargin < 4 || isempty(chunkMode)
    chunkMode = false;  % Set default value
end

if nargin < 5 || isempty(chunkSz)
    chunkSz = 1000;  % Set default value, corresponds to 1e5/100 in the origianl test
end

if nargin < 6 || isempty(gpuFlag)
    gpuFlag = false;  % Set default value
end

inpSz = size(img);



%% for testing
% d = permute(img, [3,1,2]);
% % tensor of sum of nearest neighbours
% %d = uint8(d);
% spotSize = size(conn_map,1);
% nnSum = imboxfilt3(d,[1,spotSize,spotSize],"Padding", 0, "NormalizationFactor", 1) - d;
%
% nnSum = permute(nnSum,[2,3,1]);
%spat_corr = imfilter(img,rot90(conn_map,2),'circular');
if ~chunkMode
    spat_corr = imfilter(img,rot90(conn_map,2),0);
end

%isequal(nnSum,spat_corr)
%chk = shift_array(img,[0 1 0]);
%%

% prepare empty tensor
gt = zeros([inpSz(1:2) g2range+1]);

gt = uint16(gt);


if chunkMode
    nChunk = ceil(inpSz(3)/chunkSz); % number of chunks
    maxT = inpSz(3); % max of temporal dimmention

    curGt = zeros([inpSz(1:2) g2range+1],'like', gt); % allocate before loop.

    if gpuFlag
        % move relevent vars to GPU
        conn_map = gpuArray(conn_map); 
        curGt = gpuArray(curGt);
    end

    for m = 1:nChunk
        % slice img
        curIdx = (1 + (m-1)*chunkSz) : min([chunkSz*m,maxT]);
        if gpuFlag
            curImg = gpuArray(img(:,:,curIdx));
        else
            curImg = img(:,:,curIdx);
        end
        
        curSpatCorr = imfilter(curImg,rot90(conn_map,2),0);
        
        curGt(:) = 0;
        for k = 0:g2range % loop over delays
            % nearest neighbours tensor at delay k
            t = circshift(curSpatCorr,[0,0,-k]);
            % correlate with original tensor
            curGt(:,:,k+1) = sum(curImg.*t, 3,"omitnan");
        end
        gt = gt + gather(curGt);    
    end

else % sequential
    for k = 0:g2range % loop over delays
        % nearest neighbours tensor at delay k
        t = circshift(spat_corr,[0,0,-k]);
        % correlate with original tensor
        gt(:,:,k+1) = sum(img.*t, 3,"omitnan");
    end

end

gt = double(gt);


%% Prepare outputs

G0 = squeeze(gt(:,:,1));
Ginf = squeeze(mean(gt(:,:,2:end),3));

%normalized
g0 = G0./Ginf;