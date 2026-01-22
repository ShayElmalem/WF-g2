import numpy as np


def g2_2D(img, g2range, conn_map, chunkMode=False, chunkSz=1000, gpuFlag=False):
    """
    Python port of MATLAB:
        [g0, G0, Ginf, gt] = g2_2D(img, g2range, conn_map, chunkMode, chunkSz, gpuFlag)

    Notes on exactness vs MATLAB:
    - Uses imfilter(img, rot90(conn_map,2), 0) behavior:
        * spatial correlation with a 180°-rotated kernel
        * zero-padding outside image
        * applied per-frame over (M,N) for all K
    - circshift(...,[0,0,-k]) along time axis implemented via roll by -k
    - The accumulation tensor gt is uint16, matching MATLAB casting behavior:
        * per-delay sums are computed in float64 then clipped to [0..65535] and cast to uint16
        * chunk accumulation uses uint16 addition (wraparound modulo 2^16), like MATLAB integer addition

    Parameters
    ----------
    img : (M,N,K) uint8 array, values typically 0/1
    g2range : int, maximum delay (inclusive)
    conn_map : (S,S) numeric array (kernel)
    chunkMode : bool
    chunkSz : int
    gpuFlag : bool (optional cupy/cupyx support)

    Returns
    -------
    g0 : (M,N) float64
    G0 : (M,N) float64
    Ginf : (M,N) float64
    gt : (M,N,g2range+1) float64
    """
    if img is None:
        raise ValueError("img is None")
    if g2range < 0:
        raise ValueError("g2range must be >= 0")

    # --- backend selection (numpy / cupy) ---
    xp = np
    ndcorr = None
    use_gpu = bool(gpuFlag)
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as cnd
            xp = cp
            ndcorr = cnd.correlate
        except Exception:
            # If GPU requested but not available, fall back to CPU
            xp = np
            use_gpu = False

    # --- CPU correlate (SciPy) ---
    if not use_gpu:
        try:
            from scipy.ndimage import correlate as scipy_correlate
            ndcorr = scipy_correlate
        except Exception as e:
            raise ImportError(
                "scipy is required for CPU path (scipy.ndimage.correlate). "
                "Install scipy or enable gpuFlag with cupy+cupyx available."
            ) from e

    img_x = xp.asarray(img) if use_gpu else np.asarray(img)
    if img_x.ndim != 3:
        raise ValueError(f"img must be 3D (M,N,K). Got shape {img_x.shape}")

    M, N, K = img_x.shape

    # MATLAB uses rot90(conn_map,2) then imfilter (correlation).
    # scipy/cupyx ndimage.correlate is correlation too, so we pass the rotated kernel.
    kernel = np.rot90(np.asarray(conn_map), 2)
    kernel_x = xp.asarray(kernel) if use_gpu else kernel

    # Allocate gt as uint16 (MATLAB: gt=zeros(...); gt=uint16(gt);)
    gt_u16 = np.zeros((M, N, g2range + 1), dtype=np.uint16)

    def _spatial_imfilter_zero_pad(frames_3d):
        """
        frames_3d: (M,N,T)
        returns:   (M,N,T) same dtype as correlate output (float-ish if kernel float)
        """
        # correlate over spatial dims only: use kernel (S,S,1)
        ker3 = kernel_x[:, :, None]
        # mode='constant', cval=0 implements padding with 0 like imfilter(...,0)
        return ndcorr(frames_3d, ker3, mode="constant", cval=0.0)

    def _sum_to_u16_sat(x3d):
        """
        x3d: (M,N,T) numeric (xp array)
        returns: (M,N) uint16 with MATLAB-like double->uint16 saturation.
        """
        s = xp.sum(x3d, axis=2)  # MATLAB sum(...,3,"omitnan") ; no NaNs in uint8 inputs
        # cast behavior: MATLAB double->uint16 saturates to [0..65535]
        s = xp.clip(s, 0, 65535)
        if use_gpu:
            # keep on GPU for intermediate; cast there
            return s.astype(xp.uint16)
        else:
            return s.astype(np.uint16)

    if chunkMode:
        nChunk = int(np.ceil(K / float(chunkSz)))
        maxT = K

        # curGt is uint16, like MATLAB 'like', gt
        curGt = xp.zeros((M, N, g2range + 1), dtype=xp.uint16) if use_gpu else np.zeros((M, N, g2range + 1), dtype=np.uint16)

        for m in range(nChunk):
            start = m * chunkSz
            stop = min((m + 1) * chunkSz, maxT)
            curImg = img_x[:, :, start:stop]

            curSpatCorr = _spatial_imfilter_zero_pad(curImg)

            # curGt(:)=0
            curGt[...] = 0

            for k in range(g2range + 1):
                # t = circshift(curSpatCorr,[0,0,-k])
                t = xp.roll(curSpatCorr, shift=-k, axis=2)
                # curGt(:,:,k+1) = sum(curImg.*t,3)
                curGt[:, :, k] = _sum_to_u16_sat(curImg * t)

            # gt = gt + gather(curGt)
            if use_gpu:
                # bring back chunk result and add in uint16 (wraparound)
                gt_u16 = (gt_u16 + cp.asnumpy(curGt)).astype(np.uint16, copy=False)
            else:
                gt_u16 = (gt_u16 + curGt).astype(np.uint16, copy=False)

    else:
        # spat_corr = imfilter(img,rot90(conn_map,2),0)
        spat_corr = _spatial_imfilter_zero_pad(img_x)

        # sequential loop over delays
        for k in range(g2range + 1):
            t = xp.roll(spat_corr, shift=-k, axis=2)
            plane_u16 = _sum_to_u16_sat(img_x * t)
            if use_gpu:
                gt_u16[:, :, k] = xp.asnumpy(plane_u16)
            else:
                gt_u16[:, :, k] = plane_u16

    # MATLAB: gt = double(gt);
    gt = gt_u16.astype(np.float64)

    # Outputs
    G0 = gt[:, :, 0]
    if g2range >= 1:
        Ginf = np.mean(gt[:, :, 1:], axis=2)
    else:
        # MATLAB mean over empty -> NaN; keep consistent:
        Ginf = np.full((M, N), np.nan, dtype=np.float64)

    g0 = G0 / Ginf

    return g0, G0, Ginf, gt

