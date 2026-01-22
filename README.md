# WF-g2
Wide-field photon correlation calculation


A MATLAB and Python implementation for both:

(1) straight forward  
(2) computationally efficient 

calculation of wide-field second-order photon correlation.

The code is released under the GPL-3.0 license.

This code is part of our work on Wide-field photon correlation sensing. If you find this code useful, please cite our paper:

Shay Elmalem, Gur Lubin, Michael Wayne, Claudio Bruschini, Edoardo Charbon, and Dan Oron, "Massively multiplexed wide-field photon correlation sensing," Optica 12, 451-458 (2025).
https://doi.org/10.1364/OPTICA.550498


*MATLAB*
For testing, run the test_g2.m script.

Tested in MATLAB R2021b.
MEX and MEX-CUDA files might require recompilation on your local machine.

*Python*
The python implementation is a direct translation of the MATLAB implementation using ChatGPT 5.2. 
The results were tested for bitwise accuracy compared to the MATLAB implementation. If you find any issues or discrepancies - let us know.

For testing, run the test_g2_conventional_vs_packed.py script.

Tested in Python 3.12.10.

