# LinSol: An experimental linear solver library for C ++

## Contributors
* Sokratis Anagnostopoulos
* Lingaona Zhu
* Hao Lu

## BLAS Pre-requisites
* ```OpenBLAS```
* To make sure it compiles, the path of installed OpenBLAS need to be exported first (Below is the example used in macOS)
```
brew install OpenBLAS
export LDFLAGS="-L/usr/local/opt/openblas/lib"
export CPPFLAGS="-I/usr/local/opt/openblas/include"
```
* Then in the command line compile using: 
```gcc-9 -lstdc++ -g -I/usr/local/opt/openblas/include -L/usr/local/opt/openblas/lib -lopenblas main.cpp```
* Note that feel free to use any compiler as long as it supports standard c++17;
* If using Windows please have ```OpenBLAS``` installed first, or other open libraries that include ```<cblas.h>```.
### Alternative Choice
* For macOS the ```<Accelerate/Accelerate.h>``` is pre-built as part of the framework. 
* To use it please comment out ```<cblas.h>``` in for files included it;
* And then run like below:
```gcc-9 -lstdc++ -framework Accelerate -flax-vector-conversions main.cpp ```

## Linear Solvers
* **Dense Matrix Solver**
	* Gaussian Elimination
	* LU Decomposition
	* Gauss-Seidel Iteration
	* Jacobi Iteration
	* Cholesky Factorisation
* **Sparse Matrix Solver**
	* LU Decomposition
	* Gauss-Seidel Iteration
	* Jacobi Iteration
	* Cholesky Factorisation
* **Tridiagonal (banded) Matrix**
	* Thomas Algorithm
	
## Detailed Intro for Solvers
* In our work we select to program the following methods:
### A. Direct Solvers
1. The direct Gauss elimination for dense matrices can produce quite accurate results for small systems of the order of 15 equations, but also for larger arrays when using double precision accuracy.
2. A special algorithm for tridiagonal matrices, the Thomas method [Davis, 2016], in order to handle a banded sparse matrix with a direct solver. 
3. The LU decomposition as a second direct solver for dense matrices, which belongs to the matrix factorisation methods. In this case, a special LU solver is also developed in order to handle regular or irregular sparse matrices with a direct method.
4. The Cholesky factorisation method is also implemented for symmetric matrices, which uses only half of computer memory and flops to solve a system, compared to Gauss elimination of LU decomposition methods. The right-locking procedure is adopted [Heath, 2002], and a technique to address sparse symmetric matrices is also tested.


### B. Iterative Solvers
1. Jacobi method for dense matrices and also a modified version to handle sparse matrices. The Jacobi method is very suitable for parallelisation, since the solution is updated only after completion of every iteration.
2. The Gauss Seidel method with relaxation is also examined to apply a different updating procedure, as also the under/over relaxation technique.

## Parameters
* **Regular Parameters:**
	* A the left hand side matrix
	* b the right hand side constants
* **Parameters for iterative solvers:**
	* maxit: maximum iteration time
	* tolerance: tolerance/criterion to stop the iteration
	* relaxation factor: for more stable performance

## Additional Notes
* Rewritten solvers using BLAS subroutines have no detailed comments. Please find it in its dense only version.

## References

* Davis T.A., Rajamanickam S. and Sid-Lakhdar W.M., “A survey of direct methods for sparse linear systems”, Technical Report, Department of Computer Science and Engineering, Texas A&M Univ, April 2016.
* Fausett L.V., “Applied numerical analysis using Matlab”, 2nd Editions, Pearson Eduction Inc., 2008.
Heath M.T., “Scientific computing: An introductory survey”, 2nd International Edition, McGraw-Hill, 2002.

## Contributions
* Sokratis Anagnostopoulos:
	* Implement Gauss Elimination;
	* Implement Cholesky dense/sparse; 
	* Implement Gauss-Seidel dense;
	* Implement Thomas methods; 
	* Design container tests and debugging;
	* Finish the report.
* Hao Lu: 
	* Implement LU decomposition for dense and sparse matrices; 
	* Writing the main function UI and packaging tests in testing.cpp/.h; 
	* Re-implement three dense solvers with low level BLAS functions and testings;
	* Implement functions for Matrix and CSRMatrix;
	* Managing GitHub versions and debugging;
* Lingaona Zhu:
	* Implement Gauss-seidel method for sparse matrices; 
	* Implement Jacobi method for dense and sparse matrices;
	* Generate random matrices for testing; 
	* Test solver performance and generate graphs; 
	* Compare dense and sparse solver performance;
	* Readproof: codes and report.
