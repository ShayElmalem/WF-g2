# WF-g2
Wide-Field Second-Order Photon Correlation

This repository provides MATLAB and Python implementations for computing
wide-field second-order photon correlation (g²) from binary SPAD data.

Two implementations are included:

1. Conventional (straightforward) approach
2. Computationally efficient (bit-packed / optimized) approach

Both implementations are designed to produce identical numerical results,
with the optimized version offering significant performance improvements
for large-scale datasets.

-----------------------------------------------------------------------

RELATED PUBLICATION

This code is part of our work on wide-field photon correlation sensing.
If you use this repository in your research, please cite:

Shay Elmalem, Gur Lubin, Michael Wayne, Claudio Bruschini,
Edoardo Charbon, and Dan Oron,
"Massively multiplexed wide-field photon correlation sensing,"
Optica 12, 451–458 (2025)
https://doi.org/10.1364/OPTICA.550498

-----------------------------------------------------------------------

MATLAB IMPLEMENTATION

- Reference and optimized implementations are provided.
- Includes MEX and MEX-CUDA acceleration (recompilation may be required).

Testing:
    Run the script:
        test_g2.m

Environment:
    Tested with MATLAB R2021b.
    MEX and MEX-CUDA files may need to be recompiled on your local machine.

-----------------------------------------------------------------------

PYTHON IMPLEMENTATION

- Direct translation of the MATLAB implementation.
- Generated using ChatGPT-5.2.
- Verified for bitwise-exact agreement with the MATLAB reference.

If you encounter any issues or discrepancies, please open an issue.

Testing:
    Run:
        python test_g2_conventional_vs_packed.py

Environment:
    Tested with Python 3.12.10.
    Optional GPU acceleration via CuPy (when available).

-----------------------------------------------------------------------

LICENSE

This project is released under the GPL-3.0 License.
See the LICENSE file for details.
