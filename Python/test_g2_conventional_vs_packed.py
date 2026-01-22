# test_g2_wrapper.py
import numpy as np
import cProfile
import pstats
import os

from WF_g2 import g2_2D
from WF_g2_packed import g2_2D_packed


def build_8_neighbor_conn_maps() -> list[np.ndarray]:
    base = np.zeros((3, 3), dtype=np.uint8)
    conn_maps = []

    # MATLAB: for k=1:8
    for k in range(1, 9):
        cm = base.copy()

        # MATLAB linear index (1-based, column-major), skipping center (5)
        idx = k if k < 5 else k + 1  # -> 1,2,3,4,6,7,8,9

        # Convert to (row, col) in 0-based numpy indexing, column-major
        lin0 = idx - 1
        row = lin0 % 3
        col = lin0 // 3

        cm[row, col] = 1
        conn_maps.append(cm)

    return conn_maps


def main():

    os.system('cls||clear') 

    # profile-ish settings
    reps = 10

    # random input
    M = 100
    N = 100
    K = 10_000  # packed requires multiple of 8

    if K % 8 != 0:
        raise ValueError(f"K={K} is not a multiple of 8; packed implementation requires K%8==0.")

    rng = np.random.default_rng()
    img = rng.integers(0, 2, size=(M, N, K), dtype=np.uint8)

    # params
    g2range = 20
    chunkMode = True
    chunkSz = 1000
    gpuFlag = True

    # connectivity maps (8-neighborhood, each map has exactly one '1')
    conn_maps = build_8_neighbor_conn_maps()
    nConn = len(conn_maps)

    # output prealloc (MATLAB zeros -> float64)
    gt = np.zeros((M, N, g2range + 1, nConn), dtype=np.float64)

    # straight-forward
    for _ in range(reps):
        for k in range(nConn):
            g0, G0, Ginf, gt_k = g2_2D(
                img=img,
                g2range=g2range,
                conn_map=conn_maps[k],
                chunkMode=chunkMode,
                chunkSz=chunkSz,
                gpuFlag=gpuFlag,
            )
            gt[:, :, :, k] = gt_k

    # packed (IMPORTANT: pass list via parameter name "conn_map")
    for _ in range(reps):
        g0_p, G0_p, Ginf_p, gt_p = g2_2D_packed(
            img=img,
            g2range=g2range,
            conn_map=conn_maps,      # <-- fixed
            chunkMode=chunkMode,
            chunkSz=chunkSz,
            gpuFlag=gpuFlag,
        )

    # validation (MATLAB isequal)
    for k in range(nConn):
        if np.array_equal(gt[:, :, :, k], gt_p[:, :, :, k]):
            print(f"Validation successful. The g(2) results are equal for conn_map No. {k+1}")
        else:
            print(f"Error - The g(2) results are *NOT* equal!!! for conn_map No. {k+1}")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)
