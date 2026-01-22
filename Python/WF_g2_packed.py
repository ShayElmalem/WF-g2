import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


# -----------------------------
# LUTs (CPU + GPU)
# -----------------------------
_POPCOUNT_LUT_NP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_POPCOUNT_LUT_CP = cp.asarray(_POPCOUNT_LUT_NP) if cp is not None else None


# -----------------------------
# Core helpers
# -----------------------------
def circshift(a, shift3, xp=np):
    """MATLAB circshift for 3D arrays (wrap-around)."""
    dy, dx, dt = shift3
    return xp.roll(a, shift=(dy, dx, dt), axis=(0, 1, 2))


def shift_array_fast_3d(a, shifts, xp=np):
    """
    Inline-optimized MATLAB shift_array for 3D:
      shifted = circshift(a, shifts)
      then zero-out wrapped regions  <=> zero-padded shift

    This implementation avoids roll+masking and instead does a single
    rectangular copy into a zero buffer (fastest for our use).
    """
    dy, dx, dt = shifts
    M, N, K = a.shape
    out = xp.zeros_like(a)

    # Destination ranges
    y0d = max(dy, 0)
    y1d = M + min(dy, 0)
    x0d = max(dx, 0)
    x1d = N + min(dx, 0)
    t0d = max(dt, 0)
    t1d = K + min(dt, 0)

    # Source ranges
    y0s = max(-dy, 0)
    y1s = M - max(dy, 0)
    x0s = max(-dx, 0)
    x1s = N - max(dx, 0)
    t0s = max(-dt, 0)
    t1s = K - max(dt, 0)

    if (y1d > y0d) and (x1d > x0d) and (t1d > t0d):
        out[y0d:y1d, x0d:x1d, t0d:t1d] = a[y0s:y1s, x0s:x1s, t0s:t1s]
    return out


def packBinaryArrayVectorized(binaryArray, xp=np):
    """
    Packs along time axis in consecutive groups of 8 frames:
      byte[t] packs frames [8*t .. 8*t+7]
    frame0 -> LSB, frame7 -> MSB.
    """
    a = binaryArray
    if a.dtype != xp.uint8:
        a = a.astype(xp.uint8, copy=False)

    if xp is np and a.size:
        if a.max() > 1 or a.min() < 0:
            raise ValueError("Input array must contain only binary values (0 or 1).")

    M, N, K = a.shape

    pad = (-K) % 8
    if pad:
        a = xp.concatenate([a, xp.zeros((M, N, pad), dtype=xp.uint8)], axis=2)
        K += pad

    # IMPORTANT: group consecutive frames into the last axis (bits)
    # so that a4[..., t, :] == frames [8*t .. 8*t+7]
    a4 = a.reshape(M, N, K // 8, 8)

    bitShifts = xp.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=xp.uint8).reshape(1, 1, 1, 8)

    packed = (a4 * bitShifts).sum(axis=3, dtype=xp.uint16).astype(xp.uint8, copy=False)
    return packed  # (M,N,K/8)


def sumPackedArray_LUT(packedArray, xp=np):
    """
    MATLAB sumPackedArray_LUT equivalent:
      temp = countOnesWithLUT(packedArray)  (per-byte popcount)
      sumArray = sum(temp,3)
    """
    if xp is np:
        lut = _POPCOUNT_LUT_NP
        return lut[packedArray].sum(axis=2, dtype=np.uint32)
    else:
        lut = _POPCOUNT_LUT_CP
        return lut[packedArray].sum(axis=2, dtype=cp.uint32)


def _find_single_one_rc(conn2d):
    """Return (r0,c0) 0-based location of the single '1'."""
    rc = np.argwhere(conn2d != 0)
    if rc.shape[0] != 1:
        raise ValueError("conn_map must include single neighbor only for g2_packed")
    return int(rc[0, 0]), int(rc[0, 1])


# -----------------------------
# Packed GPU fast path helper
# -----------------------------
def _sum_bitand_all_conns_gpu(packedCorr_stack, packedShiftRes, lut_cp):
    """
    packedCorr_stack: (M,N,Tp,nConn) uint8 (CuPy)
    packedShiftRes:   (M,N,Tp)       uint8 (CuPy)
    returns:          (M,N,nConn)    uint32 (CuPy)
    """
    # Broadcast AND across conn dimension
    tmp = cp.bitwise_and(packedCorr_stack, packedShiftRes[:, :, :, None])
    # popcount via LUT then sum over packed time axis
    return lut_cp[tmp].sum(axis=2, dtype=cp.uint32)


# ============================================================
# Final: g2_2D_packed (MATLAB-faithful + optimized shift + GPU fast path)
# ============================================================
def g2_2D_packed(
    img,
    g2range,
    conn_map,
    chunkMode=False,
    chunkSz=1000,
    gpuFlag=False,
):
    """
    Final, full Python port of MATLAB g2_2D_packed with:
      - Exact packBinaryArrayVectorized + sumPackedArray_LUT semantics
      - Exact shift_array semantics (zero-padded shift), inline-optimized for 3D
      - Chunk mode identical
      - Packed GPU fast path: vectorized over nConn on GPU (no inner nConn loop)

    Parameters
    ----------
    img : np.ndarray, uint8, shape (M,N,K), binary 0/1
    g2range : int, delays 0..g2range
    conn_map : 2D np.ndarray with a single 1, OR list/tuple of such arrays
    chunkMode : bool
    chunkSz : int
    gpuFlag : bool (requires cupy)

    Returns
    -------
    g0, G0, Ginf, gt (float64)
    """
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError("img must be 3D (M,N,K)")
    if img.dtype != np.uint8:
        raise ValueError("img must be uint8 (binary 0/1)")
    if img.size and (img.max() > 1 or img.min() < 0):
        raise ValueError("img must contain only 0/1 values")

    multiConnFlag = isinstance(conn_map, (list, tuple))
    if multiConnFlag:
        nConn = len(conn_map)
        if nConn == 0:
            raise ValueError("conn_map list is empty")
        # validate each has exactly one neighbor
        for cm in conn_map:
            if int(np.sum(cm)) != 1:
                raise ValueError("conn_map must include single neighbor only for g2_packed")
    else:
        nConn = 1
        if int(np.sum(conn_map)) != 1:
            raise ValueError("conn_map must include single neighbor only for g2_packed")

    use_gpu = bool(gpuFlag)
    if use_gpu and cp is None:
        raise RuntimeError("gpuFlag=True requested, but CuPy is not available/importable")

    M, N, K = img.shape

    # Allocate gt accumulator (MATLAB zeros uint32)
    if multiConnFlag:
        gt_u32 = np.zeros((M, N, g2range + 1, nConn), dtype=np.uint32)
    else:
        gt_u32 = np.zeros((M, N, g2range + 1), dtype=np.uint32)

    # ------------------------------------------------------------
    # Precompute spat_corr for sequential mode only (matches MATLAB)
    # ------------------------------------------------------------
    if not chunkMode:
        if multiConnFlag:
            spat_corr = []
            for cm in conn_map:
                r0, c0 = _find_single_one_rc(cm)
                # MATLAB: shift_array(img, [row-2, col-2, 0]) with row= r0+1 => shift = (r0-1)
                spat_corr.append(shift_array_fast_3d(img, (r0 - 1, c0 - 1, 0), xp=np))
        else:
            # MATLAB: spat_corr = imfilter(img, rot90(conn_map,2), 0)
            # For a single-1 kernel, this equals a zero-padded shift by the location of the 1 in the rotated kernel.
            krot = np.rot90(conn_map, 2)
            r2, c2 = _find_single_one_rc(krot)
            spat_corr = shift_array_fast_3d(img, (r2 - 1, c2 - 1, 0), xp=np)

    # ------------------------------------------------------------
    # Chunk mode path (MATLAB's chunkMode==true)
    # ------------------------------------------------------------
    if chunkMode:
        nChunk = int(np.ceil(K / chunkSz))
        maxT = K

        # Precompute conn shift parameters (to avoid find() inside chunk loop)
        if multiConnFlag:
            # Note: MATLAB chunk path uses ORIGINAL conn_map(:,:,k) (not rotated) with shift_array.
            shifts_rc = [(_find_single_one_rc(cm)[0] - 1, _find_single_one_rc(cm)[1] - 1) for cm in conn_map]
        else:
            # MATLAB chunk path uses imfilter(curImg, rot90(conn_map,2), 0) then pack.
            krot = np.rot90(conn_map, 2)
            r2, c2 = _find_single_one_rc(krot)
            single_shift = (r2 - 1, c2 - 1)

        # curGt exists as (M,N,g2range+1,nConn) in MATLAB chunk path
        if use_gpu:
            curGt = cp.zeros((M, N, g2range + 1, nConn), dtype=cp.uint32)
            lut = _POPCOUNT_LUT_CP
        else:
            curGt = np.zeros((M, N, g2range + 1, nConn), dtype=np.uint32)
            lut = _POPCOUNT_LUT_NP

        for m in range(nChunk):
            start = m * chunkSz
            end = min((m + 1) * chunkSz, maxT)
            if end <= start:
                continue

            curImg_cpu = img[:, :, start:end]
            xp = cp if use_gpu else np
            curImg = xp.asarray(curImg_cpu) if use_gpu else curImg_cpu

            # ---- packedCurSpatCorr ----
            if multiConnFlag:
                if use_gpu:
                    # GPU fast path: build stacked packedCorr: (M,N,Tp,nConn)
                    packedCorr_stack = []
                    for (dy, dx) in shifts_rc:
                        temp = shift_array_fast_3d(curImg, (dy, dx, 0), xp=xp)
                        packedCorr_stack.append(packBinaryArrayVectorized(temp, xp=xp))
                    packedCorr_stack = cp.stack(packedCorr_stack, axis=3)
                else:
                    packedCurSpatCorr = []
                    for (dy, dx) in shifts_rc:
                        temp = shift_array_fast_3d(curImg, (dy, dx, 0), xp=np)
                        packedCurSpatCorr.append(packBinaryArrayVectorized(temp, xp=np))
            else:
                curSpatCorr = shift_array_fast_3d(curImg, (single_shift[0], single_shift[1], 0), xp=xp)
                packedCurSpatCorr_single = packBinaryArrayVectorized(curSpatCorr, xp=xp)

            # ---- pack 0..7 remainder shifts of curImg (packedArrayShift[0..7]) ----
            packedArrayShift = [None] * 8
            packedArrayShift[0] = packBinaryArrayVectorized(curImg, xp=xp)
            for r in range(1, 8):
                temp_shft = circshift(curImg, (0, 0, r), xp=xp)
                packedArrayShift[r] = packBinaryArrayVectorized(temp_shft, xp=xp)

            # reset curGt
            if use_gpu:
                curGt.fill(0)
            else:
                curGt[...] = 0

            # ---- delays loop ----
            for delay in range(g2range + 1):
                shftVal_int = delay // 8
                shftVal_rem = delay % 8

                packedShiftRes = circshift(packedArrayShift[shftVal_rem], (0, 0, shftVal_int), xp=xp)

                if multiConnFlag:
                    if use_gpu:
                        # FAST: vectorize over nConn
                        # (M,N,nConn) uint32
                        sums = _sum_bitand_all_conns_gpu(packedCorr_stack, packedShiftRes, lut)
                        curGt[:, :, delay, :] = sums
                    else:
                        # CPU: loop per conn (still OK; can be vectorized if needed)
                        for nn in range(nConn):
                            tempPackedMultResult = np.bitwise_and(packedCurSpatCorr[nn], packedShiftRes)
                            curGt[:, :, delay, nn] = sumPackedArray_LUT(tempPackedMultResult, xp=np)
                else:
                    tempPackedMultResult = xp.bitwise_and(packedCurSpatCorr_single, packedShiftRes)
                    curGt[:, :, delay, 0] = sumPackedArray_LUT(tempPackedMultResult, xp=xp)

            # accumulate to gt_u32 on CPU (MATLAB: gt = gt + gather(curGt))
            if use_gpu:
                gt_u32 += cp.asnumpy(curGt)
            else:
                gt_u32 += curGt

        # MATLAB removes singleton conn dim in single-conn case
        if not multiConnFlag:
            gt_u32 = gt_u32[:, :, :, 0]

    # ------------------------------------------------------------
    # Sequential (no chunking) path
    # ------------------------------------------------------------
    else:
        # Pack spatial correlation(s)
        if multiConnFlag:
            packedSpatCorr = [packBinaryArrayVectorized(sc, xp=np) for sc in spat_corr]
        else:
            packedSpatCorr = packBinaryArrayVectorized(spat_corr, xp=np)

        # Pack 0..7 remainder shifts of img
        packedArrayShift = [None] * 8
        packedArrayShift[0] = packBinaryArrayVectorized(img, xp=np)
        for r in range(1, 8):
            temp_shft = circshift(img, (0, 0, r), xp=np)
            packedArrayShift[r] = packBinaryArrayVectorized(temp_shft, xp=np)

        # Optional CPU vectorization over nConn (works only for multiConnFlag)
        if multiConnFlag:
            # Stack packedSpatCorr to (M,N,Tp,nConn) for vectorized bitand+LUT
            packedCorr_stack = np.stack(packedSpatCorr, axis=3)  # uint8

        for delay in range(g2range + 1):
            shftVal_int = delay // 8
            shftVal_rem = delay % 8
            packedShiftRes = circshift(packedArrayShift[shftVal_rem], (0, 0, shftVal_int), xp=np)

            if multiConnFlag:
                # Vectorized over nConn on CPU too:
                tmp = np.bitwise_and(packedCorr_stack, packedShiftRes[:, :, :, None])
                sums = _POPCOUNT_LUT_NP[tmp].sum(axis=2, dtype=np.uint32)  # (M,N,nConn)
                gt_u32[:, :, delay, :] = sums
            else:
                tmp = np.bitwise_and(packedSpatCorr, packedShiftRes)
                gt_u32[:, :, delay] = sumPackedArray_LUT(tmp, xp=np)

    # MATLAB: gt = double(gt)
    gt = gt_u32.astype(np.float64)

    # MATLAB post outputs
    if multiConnFlag:
        G0 = np.array([])    # []
        Ginf = np.array([])  # []
        g0 = np.array([])    # []
    else:
        G0 = gt[:, :, 0]
        if g2range >= 1:
            Ginf = gt[:, :, 1:].mean(axis=2)
        else:
            Ginf = np.full((M, N), np.nan, dtype=np.float64)
        g0 = G0 / Ginf

    return g0, G0, Ginf, gt
