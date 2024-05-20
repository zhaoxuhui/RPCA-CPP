# RPCA-CPP

The CPP implementation of RPCA (Robust Principal Component Analysis) based on Armadillo, Truncated SVD, and Eigen libraries.

#### Known Issues:
* 1. The RPCA function based on Armaidillo is not very stable, where the program may not converge in some certain cases.
* 2. The RPCA function based on Truncated SVD has a risk of out of memory (OOM) when is called many times, since the created matrices are not released properly. 
