#pragma once
#include <iostream>
#include <tuple>
#include <math.h>
#include "Matrix.h"
#include "Matrix.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include <chrono>

using namespace std;

// General Parameters:
// Ax = b
// A: the input LHS Matrix
// b: the input RHS
// x: a pointer array of type T

template<class T>
void LU_dense(Matrix<T>& __restrict A, T* __restrict x, T* __restrict b);
/*LU_dense method: LU_decomposition_pp and LU_solver from ACSE-3 lecture 3 linear solvers
 Created L and P for lower matrix and identity matrix to record pivotisation;
 Pivotisation is done for better performance when dealing with non-dominant matrices;
 Lower matrix could have been stored altogether with upper one, but for the consistency of the use of backward and forward substitution with other solvers it remains separate;
 Calling the matVecMult, backward and forward substitution built in Matrix class;
 */

template<class T>
void LU_sparse(CSRMatrix<T>& __restrict A, T* __restrict x, T* __restrict b);

template<class T>
void gauss_elimination(Matrix<T>& A, T* x, T* b);

template<class T>
void jacobi_dense(Matrix<T>& A, T* x, T* b, int maxit, double tolerance);
//input matrix must be a dense matrix,two array pointers, the upper limit iteration times and one iteration tolerance
//the algorithm firstly seperate the diagonal values of A matrix from the summation, and then use other components to express each item in x array
//there is still one unknown x component on the RHS of the function, so iteration is implemented
//create one new array to store the previous solution to prevent overwriting the previous one, and put this new array as the entry of x in the next iteration

template<class T>
void jacobi_sparse(CSRMatrix<T>& A, T* x, T* b, int maxit, double tolerance);
//input matrix must be a sparse matrix,two array pointers, the upper limit iteration times and one iteration tolerance
// the algorithm is the same as jacobi_dense

template<class T>
void gauss_seidel_dense(Matrix<T>& A, T* x, T* b, int maxit, double er, double urf, int tiles);
/*
// input
A  = matrix of coefficients
b  = matrix of constants
x  = initial values for the unknown
er = termination criterion
urf = relaxation factor
tiles = pieces to slice the for loop

// output
x = solution
*/

template<class T>
void gauss_seidel_sparse(CSRMatrix<T>& A, T* x, T* b, int maxit, double tolerance);
//input matrix must be a CSRMatrix and two array pointers
//maxit: the upper limit iteration times
//tolerance: iteration tolerance
//Gauss Seidel solution is one improved method of Jacobi iteration method
//instead of using the values gained form the previous step (Jacobi method), Gauss Seidel updates the latest solution during iteration
//so it does not need a new array to store the previous solution

template<class T>
void thomas(Matrix<T>& A, T* x, T* b);

template<class T>
void cholesky_fact(Matrix<T>& A, T* x, T* b);

template<class T>
void cholesky_dense(Matrix<T>& A, T* x, T* b);

template<class T>
void cholesky_sparse(Matrix<T>& A, T* x, T* b);

template<class T>
void LU_dense_blas(Matrix<T>& __restrict A, T* __restrict x, T* __restrict b);
// Used cblas_daxpy to conduct vector operation y:= a * x + y

template<class T>
void gauss_seidel_dense_blas(Matrix<T>& A, T* x, T* b, int maxit, double er, double urf);
// Used cblas_ddot to calculate dot product

template<class T>
void jacobi_dense_blas(Matrix<T>& A, T* x, T* b, int maxit, double tolerance);
// Used cblas_ddot to calculate dot product
