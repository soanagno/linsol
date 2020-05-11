#pragma once
#include "Matrix.h"
#include <memory>

template <class T>
class CSRMatrix: public Matrix<T>
{
public:

    // constructor where we want to preallocate ourselves
    CSRMatrix(int rows, int cols, int nnzs, bool preallocate);
    // constructor where we already have allocated memory outside
    CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index);
    // destructor
    ~CSRMatrix();

    // Print out the values in our matrix
    virtual void printMatrix();
    // Print out the matrix in a dense manner
    void printDense();

    // Perform some operations with our matrix
    void matVecMult(double *input, double *output);
    // assign value of a initialised CSRMatrix using the input Matrix's values
    void fromDense(Matrix<T>& dense_in);
    // forward_substitution for CSRMatrix
    void forward_substitution(T* b, T* output);
    // backward_substitution for CSRMatrix
    void backward_substitution(T* b, T* output);

    // Explicitly using the C++11 nullptr here
    unique_ptr<int[]> row_position;
    unique_ptr<int[]> col_index;

    // How many non-zero entries we have in the matrix
    int nnzs=-1;

private:
};
