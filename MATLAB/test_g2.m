clc; clear variables; close all;

%% Test g2 functions (conventional and packed)

% Script to test the g(2) calculation code and compare the staright forward
% and packed implentations.

%% General preparations

% profile settings
profile clear
profile on

% number of repetitions (for better profiling statistics)
reps = 1;

%% Generating random input

% Spatial size
M = 100;
N = 100;

% number of binary frames
K = 1e5; % have to be a multiple of 8 for the packed version (should be handled in the future)

% Generate or load random binary matrix of size MxMxK
img = randi([0 1], M, N, K,'uint8');

%% Calc. params and flags

% number of shifts to calc g(2) [zero shift is always calculated, so the
% result will be g2range+1]
g2range = 20;

% chunk mode settings - for efficiency, the input can be dividied into
% severeal temporal chunks of size 'chunkSz'. Chunks of size 1e3 with
% g2range = 20 resulted in negligible errror
chunkMode = true;
chunkSz = 1e3;

% GPU flag for faster processing (currently supported for chunkMode only)
% 'chunkSz' should be set according to the GPU memory size
gpuFlag = false;

%% Connectivity map setting 
% setting the connectivity map (or the neighberhood) to calc. g(2). This
% exmaple is for the 8-neighberos, but every other configuration is
% possible.
% Note that for the *packed* version only a single neighbor is allowed per
% conn_map (since in the calc. is performed using bit operations)
% In addition, providing the conn_map in a cell improves performence
% significantly (packing is performed only once, and other optimizations)

base_conn_map = zeros(3); % base conn - all zeros

nConn = 8;% number of conn, can be different

conn_map_c = cell(1,nConn); % cell of all conn_maps

% loop to genereate all the 8-neighbors (center is always zero).
for k = 1:nConn
    
    conn_map_c{k} = base_conn_map; % init with zero.
    
    % set '1' to the relevent neighbor
    if k < 5
        idx = k;
    else
        idx = k+1;
    end
    conn_map_c{k}(idx) = 1;

end

%% Calc. g(2)

% alloc. output var (necessary only beacuse of loop on the straight-forward
% implenmentation. 
gt = zeros([M, N, g2range+1 nConn]);


%calc g2 - stragiht-forward func.
for m=1:reps
    for k = 1:nConn
        [g0, G0, Ginf, gt(:,:,:,k)] = g2_2D(img, g2range, conn_map_c{k}, chunkMode, chunkSz, gpuFlag);
    end
end

%calc g2 - packed func.
for m=1:reps
   [g0_p, G0_p, Ginf_p, gt_p] = g2_2D_packed(img, g2range, conn_map_c, chunkMode, chunkSz, gpuFlag);
end

%Check if the results are equal
for k = 1:nConn
    if isequal(gt(:,:,:,k), gt_p(:,:,:,k))
        disp(['Validation successful. The g(2) results are equal for conn_map No.',num2str(k)]);
    else
        disp(['Error - The g(2) results are *NOT* equal!!! for conn_map No.',num2str(k)]);
    end
end

profile off
profile report



