#include "testing.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cblas.h>
#include <memory>
#include <ctime>
#include <string>
#include <fstream>
#include <tuple>
#include <math.h>
#include <vector>

using namespace std;

template<class T>
void xinit(T* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 0;
    }
}

template<class T>
void init_condition(T* x, T* b, T* bog, int n)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
        b[i] = bog[i];
    }
}

template<class T>
void printVec(T* x, int size, bool print)
{
    if (print)
    {
        int act_size = 10;
        if (size <= 10) act_size = size;
        else cout << "(1 - 10)" << endl;
        cout << "Solution: [";
        for (int i = 0; i < act_size; i++)
        {
            cout << setprecision(8) << x[i] << " ";
        }
        cout << "]" << endl;
    }
}

template <class T>
void test_gauss_elimination(Matrix<T>& A, T* x, T* b, bool print)
{
    gauss_elimination(A, x, b);
    if (print) cout << "\nGauss Elimination Solution Dense: ";
    printVec(x, A.cols, print);
}

template<class T>
void test_LU_dense(Matrix<T>& A, T* x, T* b, bool blas, bool print)
{
    if (!blas)
    {
        LU_dense(A, x, b);
        if (print) cout << "\nLU Solution Dense: ";
    }
    else
    {
        LU_dense_blas(A, x, b);
        if (print) cout << "\nLU Solution Dense BLAS: ";
    }
    printVec(x, A.cols, print);
}

template<class T>
void test_LU_sparse(CSRMatrix<T>& A, T* x, T* b, bool print)
{
    LU_sparse(A, x, b);
    if (print) cout << "\nLU Solution Sparse: ";
    printVec(x, A.cols, print);
}

template<class T>
void test_jacobi_dense(Matrix<T>& A, T* x, T* b, int maxit, double tol, bool blas, bool print)
{
    if (!blas)
    {
        jacobi_dense(A, x, b, maxit, tol);
        if (print) cout << "\nJacobi Solution Dense: ";
    }
    else
    {
        jacobi_dense_blas(A, x, b, maxit, tol);
        if (print) cout << "\nJacobi Solution Dense BLAS: ";
    }
    printVec(x, A.cols, print);
}

template <class T>
void test_jacobi_sparse(CSRMatrix<T>& A, T* x, T* b, int maxit, double tol, bool print)
{
    jacobi_sparse(A, x, b, maxit, tol);
    if (print) cout << "\nJacobi Solution Sparse: ";
    printVec(x, A.cols, print);
}

template<class T>
void test_gauss_seidel_dense(Matrix<T>& A,T* x, T* b, int maxit, T er, T urf, bool blas, bool print)
{
    if (!blas)
    {
        gauss_seidel_dense(A, x, b, maxit, er, urf, 2);
        if (print) cout << "\nGauss-Seidel Solution Dense: ";
    }
    else
    {
        gauss_seidel_dense_blas(A, x, b, maxit, er, urf);
        if (print) cout << "\nGauss-Seidel Solution Dense BLAS: ";
    }
    printVec(x, A.cols, print);
}

template <class T>
void test_gauss_seidel_sparse(CSRMatrix<T>& A, T* x, T* b, int maxit, double tol, bool print)
{
    gauss_seidel_sparse(A, x, b, maxit, tol);
    if (print) cout << "\nGauss Seidel Solution Sparse: ";
    printVec(x, A.cols, print);
}

template <class T>
void test_thomas_tri(Matrix<T>& A, T* x, T* b, bool print)
{
    thomas(A, x, b);
    if (print) cout << "\nThomas Solution Tridiagonal: ";
    printVec(x, A.cols, print);
}

template <class T>
void test_cholesky_dense(Matrix<T>& A, T* x, T* b, bool print)
{
    cholesky_dense(A, x, b);
    if (print) cout << "\nCholesky Solution Dense: ";
    printVec(x, A.cols, print);
}

template <class T>
void test_cholesky_sparse(Matrix<T>& A, T* x, T* b, bool print)
{
    cholesky_sparse(A, x, b);
    if (print) cout << "\nCholesky Solution Sparse: ";
    printVec(x, A.cols, print);
}

template<class T>
void vectors(Matrix<T>& A, T* x_in, T* b) {
    
    // Loop counters initialization
    int i, j, k;
    int n = A.rows;
    
    // Vector initialization
    vector<double> a(n * n);
    vector<double> x(n);
    vector<double> B(n);

    // Pass the input coefficients in the selected container
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i * n + j] = A.values[i * n + j];
        }
        B[i] = b[i];
        x[i] = x_in[i];
    }


    // Preconditioning - Pivotisation
    // swap rows with ascending diagonal elements
    // from top to bottom of the matrix
    for (i = 0; i < n; i++) {
        for (k = i + 1; k < n; k++) {
            if (abs(a[i * n + i]) < abs(a[k * n + i])) {
                for (j = 0; j < n; j++) {
                    auto* temp = new double;
                    *temp = a[i * n + j]; // set temporary value before swaping
                    // Swap all row values given that the diagonal of
                    // row k is larger than the diagonal of row i
                    a[i * n + j] = a[k * n + j];
                    a[k * n + j] = *temp;
                    delete temp;
                }
                auto* temp2 = new double;
                *temp2 = b[i];
                // Similar for b
                B[i] = B[k];
                B[k] = *temp2;
                delete temp2;
            }
        }
    }


    // precondition(a, B);

    // Perform Gauss-Elimination
    for (i = 0; i < n - 1; i++) {
        for (k = i + 1; k < n; k++) {
            auto* t = new double;
            *t = a[k * n + i] / a[i * n + i];
            for (j = 0; j < n; j++) {
                // Set the coef. equal to pivot elements equal to 0 or eliminate the variables
                a[k * n + j] = a[k * n + j] - *t * a[i * n + j];
            }
            B[k] = B[k] - *t * b[i];
            delete t;
        }
    }


    for (i = n - 1; i >= 0; i--) {
        // Initialize x with the rhs of the last eq.
        x[i] = B[i];
        for (j = i + 1; j < n; j++) {
            // Subtracting all the lhs coefficient * x product of the calculated x
            if (j != i) {
                x[i] = x[i] - a[i * n + j] * x[j];
            }
        }
        // Division of rhs by the coef of the x to be calculated
        x[i] = x[i] / a[i * n + i];

    }
    cout << "Final x[n] = " << x[n - 1] << endl;
    
}

template<class T>
void vector_of_vector(Matrix<T>& A, T* x_in, T* b) {
    
    // Gauss-Seidel method using vectors of vector

    int i, j, k;
    int n = A.rows;
    
    //Vector of vector initialization
    vector<vector<double>> a(n, vector<double>(n));
    vector<double> x(n);
    vector<double> B(n);

    // Pass the input coefficients in the selected container
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = A.values[i * n + j];
        }
        B[i] = b[i];
        x[i] = x_in[i];
    }

    for (i = 0; i < n; i++) {
        for (k = i + 1; k < n; k++) {
            if (abs(a[i][i]) < abs(a[k][i])) {
                for (j = 0; j < n; j++) {
                    double temp = a[i][j];
                    a[i][j] = a[k][j];
                    a[k][j] = temp;
                }
                auto* temp2 = new double;
                *temp2 = B[i];
                B[i] = B[k];
                B[k] = *temp2;
                delete temp2;
            }
        }
    }

    for (i = 0; i < n - 1; i++) {
        for (k = i + 1; k < n; k++) {
            auto* t = new double;
            *t = a[k][i] / a[i][i];
            for (j = 0; j < n; j++) {
                a[k][j] = a[k][j] - *t * a[i][j];
            }
            B[k] = B[k] - *t * B[i];
            delete t;
        }
    }


    for (i = n - 1; i >= 0; i--) {
        x[i] = B[i];
        for (j = i + 1; j < n; j++)
            if (j != i)
                x[i] = x[i] - a[i][j] * x[j];
        x[i] = x[i] / a[i][i];

    }
    cout << "Final x[n] = " << x[n - 1] << endl;
}


template<class T>
void pointers(Matrix<T>& A, T* x_in, T* b) {

    int i, j, k;
    int n = A.rows;

    // Pointers initialization
    double* a = new double[n * n];
    double* B = new double[n];
    double* x = new double[n];

    // Pass the input coefficients in the selected container
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {

            *(a + i * n + j) = A.values[i * n + j];

        }

        *(B + i) = b[i];
        x[i] = x_in[i];

    }

    // Preconditioning
    for (i = 0; i < n; i++) {
        for (k = i + 1; k < n; k++) {
            if (abs(*(a + i * n + i)) < abs(*(a + k * n + i))) {
                for (j = 0; j < n; j++) {
                    auto* temp = new double;
                    *temp = a[i * n + j];
                    *(a + i * n + j) = a[k * n + j];
                    *(a + k * n + j) = *temp;
                    delete temp;

                }
                auto* temp2 = new double;
                *temp2 = B[i];
                *(B + i) = B[k];
                *(B + k) = *temp2;
                delete temp2;
            }
        }
    }


    for (i = 0; i < n - 1; i++) {
        for (k = i + 1; k < n; k++) {
            auto* t = new double;
            *t = *(a + k * n + i) / *(a + i * n + i);
            for (j = 0; j < n; j++) {
                *(a + k * n + j) = a[k * n + j] - *t * a[i * n + j];
            }
            *(B + k) = B[k] - *t * B[i];
            delete t;
        }
    }


    for (i = n - 1; i >= 0; i--) {

        *(x + i) = B[i];
        for (j = i + 1; j < n; j++) {
            if (j != i) {
                *(x + i) = x[i] - a[i * n + j] * x[j];
            }
        }
        *(x + i) = x[i] / a[i * n + i];
    }

    delete[] a;
    delete[] B;
    cout << "Final x[n] = " << x[n - 1] << endl;
}

template<class T>
void smart_pointer(Matrix<T>& A, T* x_in, T* b) {
    
        int i, j, k;
        int n = A.rows;
    
        // Smart Pointers initialization
        unique_ptr<double[]> a(new double[n * n]);
        unique_ptr<double[]> B(new double[n]);
        unique_ptr<double[]> x(new double[n]);

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                a[i * n + j] = A.values[i * n + j];

            }
            B[i] = b[i];
            x[i] = x_in[i];

        }

        for (i = 0; i < n; i++) {
            for (k = i + 1; k < n; k++) {
                if (abs(a[i * n + i]) < abs(a[k * n + i])) {
                    for (j = 0; j < n; j++) {
                        auto* temp = new double;
                        *temp = a[i * n + j];
                        a[i * n + j] = a[k * n + j];
                        a[k * n + j] = *temp;
                        delete temp;

                    }
                    auto* temp2 = new double;
                    *temp2 = B[i];
                    B[i] = B[k];
                    B[k] = *temp2;
                    delete temp2;
                }
            }
        }


        for (i = 0; i < n - 1; i++) {
            for (k = i + 1; k < n; k++) {
                auto* t = new double;
                *t = a[k * n + i] / a[i * n + i];
                for (j = 0; j < n; j++) {
                    a[k * n + j] = a[k * n + j] - *t * a[i * n + j];
                }
                B[k] = B[k] - *t * B[i];
                delete t;
            }
        }


        for (i = n - 1; i >= 0; i--) {
            x[i] = B[i];
            for (j = i + 1; j < n; j++) {
                if (j != i) {
                    x[i] = x[i] - a[i * n + j] * x[j];
                }
            }
            x[i] = x[i] / a[i * n + i];
        }
        cout << "Final x[n] = " << x[n - 1] << endl;
}
void time_all_rand(int n, string mat_type, double sparsity, bool print)
{
    int rows(n), cols(n);
    int maxit = 10000;
    double tol = 1e-7;
    
    Matrix<double>* dense_mat = new Matrix<double>(rows, cols, true);
    if (mat_type == "dense") dense_mat->genRanDense(true);
    else if (mat_type == "sparse") dense_mat->genRanSparse(sparsity, true);
    else if (mat_type == "tridiagonal") dense_mat->genRanTri(true);
    else
    {
        cout << "Matrix type not implemented. ";
        return;
    }
    Matrix<double>* dense_mat_hard = new Matrix<double>(*dense_mat);
    if (rows <= 20) dense_mat->printMatrix();
    
    auto* b = new double[rows * 1];
    //cout << "\nLinear system: RHS [";
    for (int i = 0; i < rows; i++)
    {
        b[i] = rand() % 10 + 1;
        //cout << b[i] << " ";
    }
    //cout << "]" << endl;
    
    auto* bog = new double[rows * 1];
    for (int i = 0; i < rows; i++)
    {
        bog[i] = b[i];
    }
    
    auto* x = new double[rows * 1];
    
    // ***************  Tests  ****************
    // gauss elimination
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c0 = new Matrix<double>(*dense_mat_hard);
    clock_t time_ge;
    time_ge = clock();
    test_gauss_elimination(*dense_mat_c0, x, b, print);
    time_ge = clock() - time_ge;
    delete dense_mat_c0;

    // LU dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c1 = new Matrix<double>(*dense_mat_hard);
    clock_t time_LU_dense;
    time_LU_dense = clock();
    test_LU_dense(*dense_mat_c1, x, b, false, print);
    time_LU_dense = clock() - time_LU_dense;
    delete dense_mat_c1;
    
    // LU dense blas
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c1b = new Matrix<double>(*dense_mat_hard);
    clock_t time_LU_dense_blas;
    time_LU_dense_blas = clock();
    test_LU_dense(*dense_mat_c1b, x, b, true, print);
    time_LU_dense_blas = clock() - time_LU_dense_blas;
    delete dense_mat_c1b;

    // jacobi dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c3 = new Matrix<double>(*dense_mat_hard);
    clock_t time_j_dense;
    time_j_dense = clock();
    test_jacobi_dense(*dense_mat_c3, x, b, maxit, tol, false, print);
    time_j_dense = clock() - time_j_dense;
    delete dense_mat_c3;

    // jacobi dense BLAS
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c4 = new Matrix<double>(*dense_mat_hard);
    clock_t time_j_dense_blas;
    time_j_dense_blas = clock();
    test_jacobi_dense(*dense_mat_c4, x, b, maxit, tol, true, print);
    time_j_dense_blas = clock() - time_j_dense_blas;
    delete dense_mat_c4;
    
    // jacobi sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c5 = new Matrix<double>(*dense_mat_hard);
    auto* sparse_mat_c5 = new CSRMatrix<double>(rows, cols, 1, true);
    sparse_mat_c5->fromDense(*dense_mat_c5);
    clock_t time_j_sparse;
    time_j_sparse = clock();
    test_jacobi_sparse(*sparse_mat_c5, x, b, maxit, tol, print);
    time_j_sparse = clock() - time_j_sparse;
    delete dense_mat_c5;
    delete sparse_mat_c5;

    // gauss_seidel dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c6 = new Matrix<double>(*dense_mat_hard);
    clock_t time_gs_dense;
    time_gs_dense = clock();
    test_gauss_seidel_dense(*dense_mat_c6, x, b, maxit, tol, 1., false, print);
    time_gs_dense = clock() - time_gs_dense;
    delete dense_mat_c6;

    // gauss_seidel dense blas
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c7 = new Matrix<double>(*dense_mat_hard);
    clock_t time_gs_dense_blas;
    time_gs_dense_blas = clock();
    test_gauss_seidel_dense(*dense_mat_c7, x, b, maxit, tol, 1., true, print);
    time_gs_dense_blas = clock() - time_gs_dense_blas;
    delete dense_mat_c7;
    
    // gauss_seidel sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c8 = new Matrix<double>(*dense_mat_hard);
    auto* sparse_mat_c8 = new CSRMatrix<double>(rows, cols, 1, true);
    sparse_mat_c8->fromDense(*dense_mat_c8);
    clock_t time_gs_sparse;
    time_gs_sparse = clock();
    test_gauss_seidel_sparse(*sparse_mat_c8, x, b, maxit, tol, print);
    time_gs_sparse = clock() - time_gs_sparse;
    delete dense_mat_c8;
    delete sparse_mat_c8;
    
    // thomas tridiagonal
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c9 = new Matrix<double>(*dense_mat_hard);
    clock_t time_th_tri;
    time_th_tri = clock();
    test_thomas_tri(*dense_mat_c9, x, b, print);
    time_th_tri = clock() - time_th_tri;
    delete dense_mat_c9;
    
    //cholesky dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c10 = new Matrix<double>(*dense_mat_hard);
    clock_t time_cho_dense;
    time_cho_dense = clock();
    test_cholesky_dense(*dense_mat_c10, x, b, print);
    time_cho_dense = clock() - time_cho_dense;
    delete dense_mat_c10;
    
    //cholesky sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c11 = new Matrix<double>(*dense_mat_hard);
    clock_t time_cho_sparse;
    time_cho_sparse = clock();
    test_cholesky_sparse(*dense_mat_c11, x, b, print);
    time_cho_sparse = clock() - time_cho_sparse;
    delete dense_mat_c11;
    
    // LU sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c2 = new Matrix<double>(*dense_mat_hard);
    auto* sparse_mat_c2 = new CSRMatrix<double>(rows, cols, 1, true);
    sparse_mat_c2->fromDense(*dense_mat_c2);
    clock_t time_LU_sparse;
    time_LU_sparse = clock();
    test_LU_sparse(*sparse_mat_c2, x, b, print);
    time_LU_sparse = clock() - time_LU_sparse;
    delete dense_mat_c2;
    delete sparse_mat_c2;
    
    // show timings
    cout << "\nTimings: " << n << " x " << n << endl;
    cout << "Sparsity: " << sparsity << endl;
    cout << setw(40) << "Gauss Elimination dense: " << setprecision(8) << (float)time_ge / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU dense: " << setprecision(8) << (float)time_LU_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU dense blas: " << setprecision(8) << (float)time_LU_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU sparse: " << setprecision(8) << (float)time_LU_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi dense: " << setprecision(8) << (float)time_j_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi dense blas: " << setprecision(8) << (float)time_j_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi sparse: " << setprecision(8) << (float)time_j_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense: " << setprecision(8) << (float)time_gs_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense blas: " << setprecision(8) << (float)time_gs_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel sparse: " << setprecision(8) << (float)time_gs_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Thomas tridiagonal: " << setprecision(8) << (float)time_th_tri / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Cholesky dense: " << setprecision(8) << (float)time_cho_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Cholesky sparse: " << setprecision(8) << (float)time_cho_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    
    
    
    delete dense_mat;
    delete dense_mat_hard;
    delete[] bog;
    delete[] x;
}

template<class T>
void time_all_given(int n, T* mat_array, T* b, bool print)
{
    int rows(n), cols(n);
    int maxit = 10000;
    double tol = 1e-7;
    Matrix<T>* dense_mat = new Matrix<T>(rows, cols, mat_array);
    Matrix<T>* dense_mat_hard = new Matrix<T>(rows, cols, mat_array);
    if (rows <= 20) dense_mat->printMatrix();
    
    auto* x = new T[rows * 1];
    auto* bog = new T[rows * 1];
    cout << "\nRHS: [";
    for (int i = 0; i < n; i++)
    {
        bog[i] = b[i];
        cout << b[i] << " ";
    }
    cout << "]" << endl;
    
    // ***************  Tests  ****************
    // gauss elimination
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c0 = new Matrix<T>(*dense_mat_hard);
    clock_t time_ge;
    time_ge = clock();
    test_gauss_elimination(*dense_mat_c0, x, b, print);
    time_ge = clock() - time_ge;
    delete dense_mat_c0;

    // LU dense
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c1 = new Matrix<T>(*dense_mat_hard);
    clock_t time_LU_dense;
    time_LU_dense = clock();
    test_LU_dense(*dense_mat_c1, x, b, false, print);
    time_LU_dense = clock() - time_LU_dense;
    delete dense_mat_c1;
    
    // LU dense blas
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c1b = new Matrix<double>(*dense_mat_hard);
    clock_t time_LU_dense_blas;
    time_LU_dense_blas = clock();
    test_LU_dense(*dense_mat_c1b, x, b, true, print);
    time_LU_dense_blas = clock() - time_LU_dense_blas;
    delete dense_mat_c1b;
    
    // LU sparse
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c2 = new Matrix<T>(*dense_mat_hard);
    auto* sparse_mat_c2 = new CSRMatrix<T>(rows, cols, 1, true);
    sparse_mat_c2->fromDense(*dense_mat_c2);
    clock_t time_LU_sparse;
    time_LU_sparse = clock();
    test_LU_sparse(*sparse_mat_c2, x, b, print);
    time_LU_sparse = clock() - time_LU_sparse;
    delete dense_mat_c2;
    delete sparse_mat_c2;

    // jacobi dense
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c3 = new Matrix<T>(*dense_mat_hard);
    clock_t time_j_dense;
    time_j_dense = clock();
    test_jacobi_dense(*dense_mat_c3, x, b, maxit, tol, false, print);
    time_j_dense = clock() - time_j_dense;
    delete dense_mat_c3;

    // jacobi dense BLAS
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c4 = new Matrix<T>(*dense_mat_hard);
    clock_t time_j_dense_blas;
    time_j_dense_blas = clock();
    test_jacobi_dense(*dense_mat_c4, x, b, maxit, tol, true, print);
    time_j_dense_blas = clock() - time_j_dense_blas;
    delete dense_mat_c4;
    
    // jacobi sparse
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c5 = new Matrix<T>(*dense_mat_hard);
    auto* sparse_mat_c5 = new CSRMatrix<T>(rows, cols, 1, true);
    sparse_mat_c5->fromDense(*dense_mat_c5);
    clock_t time_j_sparse;
    time_j_sparse = clock();
    test_jacobi_sparse(*sparse_mat_c5, x, b, maxit, tol, print);
    time_j_sparse = clock() - time_j_sparse;
    delete dense_mat_c5;
    delete sparse_mat_c5;

    // gauss_seidel dense
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c6 = new Matrix<T>(*dense_mat_hard);
    clock_t time_gs_dense;
    time_gs_dense = clock();
    test_gauss_seidel_dense(*dense_mat_c6, x, b, maxit, tol, 1., false, print);
    time_gs_dense = clock() - time_gs_dense;
    delete dense_mat_c6;

    // gauss_seidel dense blas
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c7 = new Matrix<T>(*dense_mat_hard);
    clock_t time_gs_dense_blas;
    time_gs_dense_blas = clock();
    test_gauss_seidel_dense(*dense_mat_c7, x, b, maxit, tol, 1., true, print);
    time_gs_dense_blas = clock() - time_gs_dense_blas;
    delete dense_mat_c7;
    
    // gauss_seidel sparse
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c8 = new Matrix<T>(*dense_mat_hard);
    auto* sparse_mat_c8 = new CSRMatrix<T>(rows, cols, 1, true);
    sparse_mat_c8->fromDense(*dense_mat_c8);
    clock_t time_gs_sparse;
    time_gs_sparse = clock();
    test_gauss_seidel_sparse(*sparse_mat_c8, x, b, maxit, tol, print);
    time_gs_sparse = clock() - time_gs_sparse;
    delete dense_mat_c8;
    delete sparse_mat_c8;
    
    // thomas tridiagonal
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c9 = new Matrix<T>(*dense_mat_hard);
    clock_t time_th_tri;
    time_th_tri = clock();
    test_thomas_tri(*dense_mat_c9, x, b, print);
    time_th_tri = clock() - time_th_tri;
    delete dense_mat_c9;
    
    //cholesky dense
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c10 = new Matrix<T>(*dense_mat_hard);
    clock_t time_cho_dense;
    time_cho_dense = clock();
    test_cholesky_dense(*dense_mat_c10, x, b, print);
    time_cho_dense = clock() - time_cho_dense;
    delete dense_mat_c10;
    
    //cholesky sparse
    init_condition(x, b, bog, n);
    Matrix<T>* dense_mat_c11 = new Matrix<T>(*dense_mat_hard);
    clock_t time_cho_sparse;
    time_cho_sparse = clock();
    test_cholesky_sparse(*dense_mat_c11, x, b, print);
    time_cho_sparse = clock() - time_cho_sparse;
    delete dense_mat_c11;
    
    // show timings
    cout << "\nTimings: " << endl;
    cout << setw(40) << "Gauss Elimination dense: " << setprecision(8) << (float)time_ge / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU dense: " << setprecision(8) << (float)time_LU_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU dense: " << setprecision(8) << (float)time_LU_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU sparse: " << setprecision(8) << (float)time_LU_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi dense: " << setprecision(8) << (float)time_j_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi dense blas: " << setprecision(8) << (float)time_j_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi sparse: " << setprecision(8) << (float)time_j_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense: " << setprecision(8) << (float)time_gs_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense blas: " << setprecision(8) << (float)time_gs_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel sparse: " << setprecision(8) << (float)time_gs_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Thomas tridiagonal: " << setprecision(8) << (float)time_th_tri / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Cholesky dense: " << setprecision(8) << (float)time_cho_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Cholesky sparse: " << setprecision(8) << (float)time_cho_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    
    delete dense_mat_hard;
    delete[] x;
    delete[] bog;
}

void time_dense_vs_sparse(int n, string mat_type, double sparsity, bool print)
{
//    if ((n >= 300) && (n < 600))
//    {
//        cout << "For the implementation of LU sparse it costs long time to run it when the size is larger than 300. Type in 0/1 to continue/abort." << endl;
//        bool go_on;
//        cin >> go_on;
//    }
//    else if (n >= 600)
//    {
//        cout << "Matrix too large to get solution in hours so to save your time I will abort. ";
//        return;
//    }
    int rows(n), cols(n);
    int maxit = 10000;
    double tol = 1e-7;
    
    Matrix<double>* dense_mat = new Matrix<double>(rows, cols, true);
    if (mat_type == "dense") dense_mat->genRanDense(true);
    else if (mat_type == "sparse") dense_mat->genRanSparse(sparsity, true);
    else if (mat_type == "tridiagonal") dense_mat->genRanTri(true);
    else
    {
        cout << "Matrix type not implemented. ";
        return;
    }
    Matrix<double>* dense_mat_hard = new Matrix<double>(*dense_mat);
    if (rows <= 20) dense_mat->printMatrix();
    
    auto* b = new double[rows * 1];
    //cout << "\nLinear system: RHS [";
    for (int i = 0; i < rows; i++)
    {
        b[i] = rand() % 10 + 1;
        //cout << b[i] << " ";
    }
    //cout << "]" << endl;
    
    auto* bog = new double[rows * 1];
    for (int i = 0; i < rows; i++)
    {
        bog[i] = b[i];
    }
    
    auto* x = new double[rows * 1];
    
    // ***************  Tests  ****************
    // LU dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c1 = new Matrix<double>(*dense_mat_hard);
    clock_t time_LU_dense;
    time_LU_dense = clock();
    test_LU_dense(*dense_mat_c1, x, b, false, print);
    time_LU_dense = clock() - time_LU_dense;
    delete dense_mat_c1;

    // jacobi dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c3 = new Matrix<double>(*dense_mat_hard);
    clock_t time_j_dense;
    time_j_dense = clock();
    test_jacobi_dense(*dense_mat_c3, x, b, maxit, tol, false, print);
    time_j_dense = clock() - time_j_dense;
    delete dense_mat_c3;
    
    // jacobi sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c5 = new Matrix<double>(*dense_mat_hard);
    auto* sparse_mat_c5 = new CSRMatrix<double>(rows, cols, 1, true);
    sparse_mat_c5->fromDense(*dense_mat_c5);
    clock_t time_j_sparse;
    time_j_sparse = clock();
    test_jacobi_sparse(*sparse_mat_c5, x, b, maxit, tol, print);
    time_j_sparse = clock() - time_j_sparse;
    delete dense_mat_c5;
    delete sparse_mat_c5;

    // gauss_seidel dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c6 = new Matrix<double>(*dense_mat_hard);
    clock_t time_gs_dense;
    time_gs_dense = clock();
    test_gauss_seidel_dense(*dense_mat_c6, x, b, maxit, tol, 1., false, print);
    time_gs_dense = clock() - time_gs_dense;
    delete dense_mat_c6;
    
    // gauss_seidel sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c8 = new Matrix<double>(*dense_mat_hard);
    auto* sparse_mat_c8 = new CSRMatrix<double>(rows, cols, 1, true);
    sparse_mat_c8->fromDense(*dense_mat_c8);
    clock_t time_gs_sparse;
    time_gs_sparse = clock();
    test_gauss_seidel_sparse(*sparse_mat_c8, x, b, maxit, tol, print);
    time_gs_sparse = clock() - time_gs_sparse;
    delete dense_mat_c8;
    delete sparse_mat_c8;
    
    //cholesky dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c10 = new Matrix<double>(*dense_mat_hard);
    clock_t time_cho_dense;
    time_cho_dense = clock();
    test_cholesky_dense(*dense_mat_c10, x, b, print);
    time_cho_dense = clock() - time_cho_dense;
    delete dense_mat_c10;
    
    //cholesky sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c11 = new Matrix<double>(*dense_mat_hard);
    clock_t time_cho_sparse;
    time_cho_sparse = clock();
    test_cholesky_sparse(*dense_mat_c11, x, b, print);
    time_cho_sparse = clock() - time_cho_sparse;
    delete dense_mat_c11;
    
    // LU sparse
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c2 = new Matrix<double>(*dense_mat_hard);
    auto* sparse_mat_c2 = new CSRMatrix<double>(rows, cols, 1, true);
    sparse_mat_c2->fromDense(*dense_mat_c2);
    clock_t time_LU_sparse;
    time_LU_sparse = clock();
    if (n < 600)
    {
        test_LU_sparse(*sparse_mat_c2, x, b, print);
    }
    time_LU_sparse = clock() - time_LU_sparse;
    delete dense_mat_c2;
    delete sparse_mat_c2;
    
    // show timings
    cout << "\nTimings: " << n << " x " << n << endl;
    cout << "\nSparsity: " << sparsity << endl;
    cout << setw(40) << "LU dense: " << setprecision(8) << (float)time_LU_dense / CLOCKS_PER_SEC << " seconds" << endl;
    if (n < 600) {cout << setw(40) << "LU sparse: " << setprecision(8) << (float)time_LU_sparse / CLOCKS_PER_SEC << " seconds" << endl;}
    cout << setw(40) << "Jacobi dense: " << setprecision(8) << (float)time_j_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi sparse: " << setprecision(8) << (float)time_j_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense: " << setprecision(8) << (float)time_gs_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel sparse: " << setprecision(8) << (float)time_gs_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Cholesky dense: " << setprecision(8) << (float)time_cho_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Cholesky sparse: " << setprecision(8) << (float)time_cho_sparse / CLOCKS_PER_SEC << " seconds" << endl;
    
    delete dense_mat;
    delete dense_mat_hard;
    delete[] bog;
    delete[] x;
}

void time_dense_vs_blas(int n, bool print)
{
    int rows(n), cols(n);
    int maxit = 10000;
    double tol = 1e-7;
    
    Matrix<double>* dense_mat = new Matrix<double>(rows, cols, true);
    dense_mat->genRanDense(true);
    Matrix<double>* dense_mat_hard = new Matrix<double>(*dense_mat);
    if (rows <= 20) dense_mat->printMatrix();
    
    auto* b = new double[rows * 1];
    for (int i = 0; i < rows; i++)
    {
        b[i] = rand() % 10 + 1;
    }
    auto* bog = new double[rows * 1];
    for (int i = 0; i < rows; i++)
    {
        bog[i] = b[i];
    }
    
    auto* x = new double[rows * 1];
    
    // ***************  Tests  ****************
    // LU dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c1 = new Matrix<double>(*dense_mat_hard);
    clock_t time_LU_dense;
    time_LU_dense = clock();
    test_LU_dense(*dense_mat_c1, x, b, false, print);
    time_LU_dense = clock() - time_LU_dense;
    delete dense_mat_c1;
    
    // LU dense blas
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c1b = new Matrix<double>(*dense_mat_hard);
    clock_t time_LU_dense_blas;
    time_LU_dense_blas = clock();
    test_LU_dense(*dense_mat_c1b, x, b, true, print);
    time_LU_dense_blas = clock() - time_LU_dense_blas;
    delete dense_mat_c1b;

    // jacobi dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c3 = new Matrix<double>(*dense_mat_hard);
    clock_t time_j_dense;
    time_j_dense = clock();
    test_jacobi_dense(*dense_mat_c3, x, b, maxit, tol, false, print);
    time_j_dense = clock() - time_j_dense;
    delete dense_mat_c3;
    
    // jacobi dense BLAS
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c4 = new Matrix<double>(*dense_mat_hard);
    clock_t time_j_dense_blas;
    time_j_dense_blas = clock();
    test_jacobi_dense(*dense_mat_c4, x, b, maxit, tol, true, print);
    time_j_dense_blas = clock() - time_j_dense_blas;
    delete dense_mat_c4;

    // gauss_seidel dense
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c6 = new Matrix<double>(*dense_mat_hard);
    clock_t time_gs_dense;
    time_gs_dense = clock();
    test_gauss_seidel_dense(*dense_mat_c6, x, b, maxit, tol, 1., false, print);
    time_gs_dense = clock() - time_gs_dense;
    delete dense_mat_c6;
    
    // gauss_seidel dense blas
    init_condition(x, b, bog, n);
    Matrix<double>* dense_mat_c7 = new Matrix<double>(*dense_mat_hard);
    clock_t time_gs_dense_blas;
    time_gs_dense_blas = clock();
    test_gauss_seidel_dense(*dense_mat_c7, x, b, maxit, tol, 1., true, print);
    time_gs_dense_blas = clock() - time_gs_dense_blas;
    delete dense_mat_c7;
    
    // show timings
    cout << "\nTimings: " << endl;
    cout << "Size: " << n << " x " << n << endl;
    cout << setw(40) << "LU dense: " << setprecision(8) << (float)time_LU_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "LU dense blas: " << setprecision(8) << (float)time_LU_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi dense: " << setprecision(8) << (float)time_j_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Jacobi dense blas: " << setprecision(8) << (float)time_j_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense: " << setprecision(8) << (float)time_gs_dense / CLOCKS_PER_SEC << " seconds" << endl;
    cout << setw(40) << "Gauss-Seidel dense blas: " << setprecision(8) << (float)time_gs_dense_blas / CLOCKS_PER_SEC << " seconds" << endl;
    
    delete dense_mat;
    delete dense_mat_hard;
    delete[] bog;
    delete[] x;
}

void compare_results()
{
    double* mat_array = new double[15 * 15]
    {98,   0,   0,   0,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,  91,   0,  13,   0,   0,   0,   0,   0,   5,   6,  6,   0,   0,   0,
    0,   0, 108,   0,   0,   0,   0,   0,   0,   0,   5,   0,  13,   0,   0,
    0,  13,   0, 145,   0,   0,   0,   0,   0,   0,   0,  10,  11,   0,   6,
    5,   0,   0,   0, 118,   0,   8,  11,   0,   9,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0, 133,   0,   0,  11,  12,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   8,   0, 142,  13,   0,   0,  11,   0,   6,   0,   0,
    0,   0,   0,   0,  11,   0,  13,  82,   0,   0,   0,   5,   0,   0,  10,
    0,   0,   0,   0,   0,  11,   0,   0,  93,   0,   0,   7,   0,   5,   0,
    0,   5,   0,   0,   9,  12,   0,   0,   0,  72,   0,   9,   0,   0,   0,
    0,   6,   5,   0,   0,   0,  11,   0,   0,   0, 114,   0,   0,   0,   0,
    0,   6,   0,  10,   0,   0,   0,   5,   7,   9,   0,  52,  11,   6,   0,
    0,   0,  13,  11,   0,   0,   6,   0,   0,   0,   0,  11, 106,   9,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   5,   0,   0,   6,   9, 116,   7,
        0,   0,   0,   6,   0,   0,   0,  10,   0,   0,   0,   0,   0,   7,  58};
    double* b = new double[15]{5, 2, 6, 5, 10, 8, 3, 8, 6, 4, 1, 1, 3, 10, 1};
    cout << "Showing the result of pre-set Matrix and RHS: " << endl;
    time_all_given(15, mat_array, b, true);
    cout << "Thomas solution has been verified from multiple random tridiagonal solutions. The difference showed here is due to the non-tridiagonal matrix. The difference is also fairly small because the diagonal entries for this matrix is set to be dominant." << endl;
}

void compare_containers()
{
    cout << "Please type in two integers to determine the start and end matrix size, separated by a blank space. The size will be multiplied by a factor of 2 (dimensions expressed as 10 * 2 ^ power: \n";
    int power_start, power_end;
    cin >> power_start >> power_end;
    
    cout << "Press any key to continue. " << endl;
    for (int i = power_start; i <= power_end; i++)
    {
        // Initializations
        const int no = 10 * pow(2, i);  // Dimensions
        const int rows(no), cols(no);
        const int n(no);
        double* x = new double[n];  // Initialization of x

        auto* A = new Matrix<double>(rows, cols, true);
        A->genRanSparse(0.7, true);
        Matrix<double>* A_back = new Matrix<double>(*A);
        
        // Set b values
        auto* b = new double[n];
        for (int i = 0; i < rows; i++)
        {
            b[i] = rand() % 10 + 1;
        }
        auto* bog = new double[rows * 1];
        for (int i = 0; i < rows; i++)
        {
            bog[i] = b[i];
        }
        
        cout << "Comparison begins with matrix size of: " << no << " x " << no << endl;
        cout << "-------------------------------------------------" << endl;
        if (n > 1000) cout << "This will take a while. " << endl;
        clock_t time_req;

        // Main function gaussian elimination solver
        init_condition(x, b, bog, n);
        Matrix<double>* A1 = new Matrix<double>(*A_back);
        time_req = clock();
        gauss_elimination(*A1, x, b);
        time_req = clock() - time_req;
        cout << "Smart pointers: " << (double)time_req / CLOCKS_PER_SEC << " seconds" << endl << endl;
        delete A1;

        // Vector container
        init_condition(x, b, bog, n);
        Matrix<double>* A2 = new Matrix<double>(*A_back);
        time_req = clock();
        vectors(*A2, x, b);
        cout << "Vector time: " << (double)time_req / CLOCKS_PER_SEC << " seconds" << endl << endl;
        delete A2;

        // Vector of vector container
        init_condition(x, b, bog, n);
        Matrix<double>* A3 = new Matrix<double>(*A_back);
        time_req = clock();
        vector_of_vector(*A3, x, b);
        cout << "Vector of vector time: " << (double)time_req / CLOCKS_PER_SEC << " seconds" << endl << endl;
        delete A3;

        // Pointer container
        init_condition(x, b, bog, n);
        Matrix<double>* A4 = new Matrix<double>(*A_back);
        time_req = clock();
        pointers(*A4, x, b);
        time_req = clock() - time_req;
        cout << "Pointers: " << (double)time_req / CLOCKS_PER_SEC << " seconds" << endl << endl;
        delete A4;

        // Smart pointer container
        init_condition(x, b, bog, n);
        Matrix<double>* A5 = new Matrix<double>(*A_back);
        time_req = clock();
        smart_pointer(*A5, x, b);
        time_req = clock() - time_req;
        cout << "Smart pointer time: " << (double)time_req / CLOCKS_PER_SEC << " seconds" << endl << endl;
        delete A5;

        delete A;
        delete[] b;
        delete[] x;
    }
}
