#pragma once
#include <iostream>
#include <memory>
#include <ctime>
#include "Matrix.h"
#include "CSRMatrix.h"
#include "solver.h"
#include "solver.cpp"
#include <algorithm>
#include <iomanip>

using namespace std;

// initialise x
template<class T>
void xinit(T* x, int n);

// initialise b, x
template<class T>
void init_condition(T* x, T* b, T* bog, int n);

// print solution depends on size: up to 10
template<class T>
void printVec(T* x, int size, bool print);

// package to test solvers
template<class T>
void test_LU_dense(Matrix<T>& A, T* b, bool blas, bool print);

template<class T>
void test_LU_sparse(CSRMatrix<T>& A, T* b, bool print);

template <class T>
void test_gauss_elimination(Matrix<T>& A, T* b, bool print);

template<class T>
void test_jacobi_dense(Matrix<T>& A, T* b, int maxit, double tol, bool blas, bool print);

template <class T>
void test_jacobi_sparse(CSRMatrix<T>& A, T* b, int maxit, double tol, bool print);

template<class T>
void test_gauss_seidel_dense(Matrix<T>& A, T* b, int maxit, T er, T urf, bool blas, bool print);

template <class T>
void test_gauss_seidel_sparse(CSRMatrix<T>& A, T* b, int maxit, double tol, bool print);

template<class T>
void test_thomas_tri(Matrix<T>& A, T* b, bool print);

template <class T>
void test_cholesky_dense(Matrix<T>& A, T* b, bool print);

template <class T>
void test_cholesky_sparse(Matrix<T>& A, T* b, bool print);

// Gaussian elimination with different containers: vectors
template<class T>
void vectors(Matrix<T>& A, T* x_in, T* b);

// Gaussian elimination with different containers: vectors
template<class T>
void vector_of_vector(Matrix<T>& A, T* x_in, T* b);

// Gaussian elimination with different containers: vectors
template<class T>
void pointers(Matrix<T>& A, T* x_in, T* b);

// Gaussian elimination with different containers: vectors
template<class T>
void smart_pointer(Matrix<T>& A, T* x_in, T* b);

// test timings with randomly generated matrices
void time_all_rand(int n, string mat_type, bool print);

// if user want to specify a matrix to test call this function
template<class T>
void time_all_given(T* mat_array, T* b, bool);

// timings for all dense (no blas) and sparse solvers
void time_dense_vs_sparse(int n, string mat_type, double sparsity, bool print);

// timings to compare dense and blas
void time_dense_vs_blas(int n, bool print);

// using a specified array to run all solvers to show they have the same results
// will print 10 of the solutions
void compare_results();

// compare containers
void compare_containter();
