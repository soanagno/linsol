#pragma once
#include <iostream>
#include "CSRMatrix.h"
#include <cmath>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <vector>

// Constructor - using an initialisation list here
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate): Matrix<T>(rows, cols, false), nnzs(nnzs)
{
   // If we don't pass false in the initialisation list base constructor, it would allocate values to be of size
   // rows * cols in our base matrix class
   // So then we need to set it to the real value we had passed in
   this->preallocated = preallocate;

   // If we want to handle memory ourselves
   if (this->preallocated)
   {
      // Must remember to delete this in the destructor
      this->values.reset(new T[this->nnzs]);
      this->row_position.reset(new int[this->rows+1]);
      this->col_index.reset(new int[this->nnzs]);
   }
}

// Constructor - now just setting the value of our T pointer
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index): Matrix<T>(rows, cols, values_ptr), nnzs(nnzs)
{
    this->row_position.reset(row_position);
    this->col_index.reset(col_index);
}

// destructor
// unique pointers used so they will take care of the memory
template <class T>
CSRMatrix<T>::~CSRMatrix() {}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void CSRMatrix<T>::printMatrix() 
{ 
   std::cout << "Printing matrix: sparse" << std::endl;
   std::cout << "Values: ";
   for (int j = 0; j< this->nnzs; j++)
   {  
      std::cout << this->values[j] << " ";      
   }
   std::cout << std::endl;
   std::cout << "row_position: ";
   for (int j = 0; j< this->rows+1; j++)
   {  
      std::cout << this->row_position[j] << " ";      
   }
   std::cout << std::endl;   
   std::cout << "col_index: ";
   for (int j = 0; j< this->nnzs; j++)
   {  
      std::cout << this->col_index[j] << " ";      
   }
   std::cout << std::endl << std::endl;  
}

// Do a matrix-vector product
// output = this * input
template<class T>
void CSRMatrix<T>::matVecMult(double *input, double *output)
{
   if (input == nullptr || output == nullptr)
   {
      std::cerr << "Input or output haven't been created" << std::endl;
      return;
   }

   // Set the output to zero
   for (int i = 0; i < this->rows; i++)
   {
      output[i] = 0.0;
   }

   int val_counter = 0;
   // Loop over each row
   for (int i = 0; i < this->rows; i++)
   {
      // Loop over all the entries in this col
      for (int val_index = this->row_position[i]; val_index < this->row_position[i+1]; val_index++)
      {
         // This is an example of indirect addressing
         // Can make it harder for the compiler to vectorise!
         output[i] += this->values[val_index] * input[this->col_index[val_index]];

      }
   }
}

template<class T>
void CSRMatrix<T>::fromDense(Matrix<T>& dense_in)
{
    int m = dense_in.rows;
    int n = dense_in.cols;
    this->rows = m;
    this->cols = n;
    this->row_position.reset(new int[m + 1]);
    
    // calculating the row position for given values
    int counter = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (dense_in.values[i * n + j] != 0) counter++;
        }
        this->row_position[i + 1] = counter;
    }
    // reset rowposition[0] which is out of range for length dense_in.rows
    this->row_position[0] = 0;
    // resize the nnzs
    this->nnzs = this->row_position[m];
    
    vector<T> temp_value;
    vector<int> temp_col;
    
    this->values.reset(new T[this->nnzs]);
    this->col_index.reset(new int[this->nnzs]);

    // store values of input dense matrix temporarily
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (dense_in.values[i * n + j] != 0)
            { temp_value.push_back(dense_in.values[i * n + j]);
                temp_col.push_back(j);
            }
        }
    }
    // assign stored cols and values to current CSRMatrix
    for (int i = 0; i < this->nnzs; i++)
    {
        this->values[i] = temp_value[i];
        this->col_index[i] = temp_col[i];
    }
}

template<class T>
void CSRMatrix<T>::forward_substitution(T* b, T* output)
{
    for (int r = 0; r < this->rows; r++)
    {
        int start_index = this->row_position[r];
        int end_index;
        for (int i = start_index; i < this->row_position[r + 1]; i++)
        {
            if (this->col_index[i] == r)
            {
                end_index = i;
            }
        }
        T val;
        int col;
        double s(0);
        for (int k = start_index; k < end_index; k++)
        {
            col = this->col_index[k];
            if (this->col_index[k] != r) val = this->values[k];
            else val = 1;
            for (int c = 0; c < this->rows; c++)
            {
                if (c == col) s += val * output[c];
            }
        }
        output[r] = (b[r] - s) / 1;
    }
}

template<class T>
void CSRMatrix<T>::backward_substitution(T* b, T* output)
{
    for (int r = this->rows; r > 0; r--)
    {
        int start_index;
        int end_index = this->row_position[r] - 1;
        for (int i = end_index; i >= this->row_position[r - 1]; i--)
        {
            if (this->col_index[i] == r - 1)
            {
                start_index = i;
            }
        }
        T val;
        int col;
        double s(0);
        for (int k = start_index + 1; k <= end_index; k++)
        {
            col = this->col_index[k];
            val = this->values[k];
            for (int c = 0; c < this->rows; c++)
            {
                if (c == col) s += val * output[c];
            }
        }
        output[r - 1] = (b[r - 1] - s) / this->values[start_index];
    }
}

