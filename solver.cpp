#pragma once
#include "solver.h"
#include <memory>
#include <cblas.h>
#include <chrono>

template<class T>
void LU_dense(Matrix<T>& __restrict A, T* __restrict x, T* __restrict b)
{
    // check if a square matrix
    if (A.rows != A.cols)
    {
        cerr << "Cannot decompose non-square matrix into LU. \n";
        return;
    }

    const int m = A.cols;
    auto* L = new Matrix<T>(m, m, true);
    auto* P_ = new Matrix<T>(m, m, true);
	// create an empty matrix L
	for (int i = 0; i < m * m; i++)
	{
		L->values[i] = 0;
	}
	// create eye matrix P_
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i == j) { P_->values[i * m + j] = 1; }
			else P_->values[i * m + j] = 0;
		}
	}
    // LU decomposition
    // The algorithm is from ACSE-3 Lecture linear solvers: LU_decomposition_pp
	for (int k = 0; k < m - 1; k++)
	{
        // partial pivotisation
        // find max value in the current col
		int index = k * A.cols + k;
        for (int i = k; i < A.cols; i++)
        {
            if (abs(A.values[i * A.cols + k]) > abs(A.values[index]))
            {
                index = i * A.cols + k;
            }
        }
        int j = index / A.cols;
        
        // swap the rows for all three matrices
        // this can be further simplified but requires corresponding changes
        // of backward & forward substitution, which will cause inconsistency
        // for cholesky methods
        for (int i = 0; i < A.cols; i++)
        {
            T* numA = new T;
            *numA = A.values[k * A.rows + i];
            A.values[k * A.rows + i] = A.values[j * A.rows + i];
            A.values[j * A.rows + i] = *numA;
            
            T* numP_ = new T;
            *numP_ = P_->values[k * P_->rows + i];
            P_->values[k * P_->rows + i] = P_->values[j * P_->rows + i];
            P_->values[j * P_->rows + i] = *numP_;
            
            T* numL = new T;
            *numL = L->values[k * L->rows + i];
            L->values[k * L->rows + i] = L->values[j * L->rows + i];
            L->values[j * L->rows + i] = *numL;
            
            delete numA;
            delete numP_;
            delete numL;
        }
        // loop over resting rows to conduct vector daxpy
        // find the constant divisor for the row: s
        // then each row below y := -s * x + y
        // where x is the current row
        // and y is rows below the current row
		for (int i = k + 1; i < m; i++)
		{
			const double s = A.values[i * m + k] / A.values[k * m + k];
            // store the s in lower matrix
			L->values[i * m + k] = s;
			for (int j = k; j < m; j++)
			{
				A.values[i * m + j] -= A.values[k * m + j] * s;
			}
		}
	}
    
	// add diagnal eye into L
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i == j) { L->values[i * m + j] = 1; }
		}
	}
    
    // now A becomes upper triangle, L the lower triangle and P_ the swapping reference matrix
    // now to solve L(Ux) = P^-1 @ b
    auto* pinvb = new double[m * 1];
    // get pinvb = P^-1 @ b
    P_->matVecMult(b, pinvb);
    auto* y = new double[m * 1];
    // forward substitution to get Ly = pinvb
    L->forward_sub(y, pinvb);
    // backward substitution to get Ux = y
    A.backward_sub(x, y);

    delete L;
    delete P_;
    delete[] pinvb;
    delete[] y;
}

template<class T>
void LU_sparse(CSRMatrix<T>& __restrict A, T* __restrict x, T* __restrict b)
{
    // create a identity matrix
    auto* p_col = new int[A.rows * 1];
    for (int c = 0; c < A.rows; c++)
    {
        p_col[c] = c;
    }
    // same algorithm as LU_dense from ACSE-3 Lecture 3 Linear solvers
    for (int i = 0; i < A.rows; i++)
    {
        int nnzs_current = A.row_position[i + 1] - A.row_position[i];

        // for each diagonal entry:
        // find the largest row index below
        // loop over col_index and corresponding values
        double max_value(0);
        int max_value_index(i);
        int nnzs_max_row;
        int max_value_start_index;
        int current_start_index = A.row_position[i];

        // find max value and its index in the value array
        for (int k = A.row_position[i]; k < A.nnzs; k++)
        {
            if (A.col_index[k] == i)
            {
                if (abs(A.values[k]) > abs(max_value))
                {
                    max_value_index = k;
                    max_value = A.values[k];
                }
            }
        }
        
        // simulate row swapping for identity matrix
        for (int m = 0; m < A.rows; m++)
        {
            if ((max_value_index >= A.row_position[m]) && (max_value_index < A.row_position[m + 1]))
            {
                unique_ptr<int> temp_p_col(new int);
                *temp_p_col = p_col[m];
                p_col[m] = p_col[i];
                p_col[i] = *temp_p_col;
            }
        }

        // as the matrix is diagonal positive
        // dont need to worry about 0s on the diagnal after partial pivoting
        // find the start index of the row of where the max value was found and how many non-zeros in that row
        for (int j = 0; j < A.rows; j++)
        {
            if ((max_value_index >= A.row_position[j]) & (max_value_index < A.row_position[j + 1]))
            {
                max_value_start_index = A.row_position[j];
                nnzs_max_row = A.row_position[j + 1] - A.row_position[j];
            }
        }
        
        // create pointers to store current row and max value row non-zeros and their col indices;
        unique_ptr<double[]> temp_current_value(new double[nnzs_current]);
        unique_ptr<int[]> temp_current_col(new int[nnzs_current]);
        unique_ptr<double[]> temp_max_value(new double[nnzs_max_row]);
        unique_ptr<int[]> temp_max_col(new int[nnzs_max_row]);

        int nnzs_in_between = max_value_start_index - current_start_index - nnzs_current;
        int in_between_start_index = current_start_index + nnzs_current;
        
        // temporarily store current row non-zeros value/col
        for (int in = 0; in < nnzs_current; in++)
        {
            temp_current_value[in] = A.values[current_start_index + in];
            temp_current_col[in] = A.col_index[current_start_index + in];
        }

        // temporarily store max value row non-zeros value/col
        for (int in = 0; in < nnzs_max_row; in++)
        {
            temp_max_value[in] = A.values[max_value_start_index + in];
            temp_max_col[in] = A.col_index[max_value_start_index + in];
        }

        // check if the rows need to be changed
        // if needed the swapped row should be ahead of the swapping row
        if (max_value_start_index > current_start_index)
        {
            // check if there is in between value, if there is:
            if (in_between_start_index != max_value_start_index)
            {
                // create pointers to store temp value/col for in between elements
                unique_ptr<double[]> temp_between_value(new double[nnzs_in_between]);
                unique_ptr<int[]> temp_between_col(new int[nnzs_in_between]);
                // store in between value/col
                for (int in = 0; in < nnzs_in_between; in++)
                {
                    temp_between_value[in] = A.values[in_between_start_index + in];
                    temp_between_col[in] = A.col_index[in_between_start_index + in];
                }

                // refill value/col: max value row -> current row
                for (int ivc = 0; ivc < nnzs_max_row; ivc++)
                {
                    A.values[current_start_index + ivc] = temp_max_value[ivc];
                    A.col_index[current_start_index + ivc] = temp_max_col[ivc];
                }

                // refill value/col: change pos of in between ones
                for (int ivc = 0; ivc < nnzs_in_between; ivc++)
                {
                    A.values[current_start_index + nnzs_max_row + ivc] = temp_between_value[ivc];
                    A.col_index[current_start_index + nnzs_max_row + ivc] = temp_between_col[ivc];
                }

                // refill value/col: current now behind in between ones
                for (int ivc = 0; ivc < nnzs_current; ivc++)
                {
                    A.values[current_start_index + nnzs_max_row + nnzs_in_between + ivc] = temp_current_value[ivc];
                    A.col_index[current_start_index + nnzs_max_row + nnzs_in_between + ivc] = temp_current_col[ivc];
                }

                // if there is a difference between nnzs of swapping two rows
                // if not, no need to change row position
                if (nnzs_current != nnzs_max_row)
                {
                    int end_index;
                    for (int o = 0; o < A.rows; o++)
                    {
                        if ((A.row_position[o + 1] > max_value_index) && (A.row_position[o] <= max_value_index))
                        {
                            end_index = A.row_position[o + 1];
                        }
                    }
                    int nnzs_diff = nnzs_max_row - nnzs_current;
                    for (int irp = i + 1; irp < A.rows; irp++)
                    {
                        if (A.row_position[irp] < end_index)
                        {
                            A.row_position[irp] += nnzs_diff;
                        }
                        else break;
                    }
                }
            }
            
            // if no elements in between
            else
            {
                // refill values/col: max value -> current
                for (int ivc = 0; ivc < nnzs_max_row; ivc++)
                {
                    A.values[current_start_index + ivc] = temp_max_value[ivc];
                    A.col_index[current_start_index + ivc] = temp_max_col[ivc];
                }

                // refill values/col: current -> max value
                for (int ivc = 0; ivc < nnzs_current; ivc++)
                {
                    A.values[current_start_index + nnzs_max_row + ivc] = temp_current_value[ivc];
                    A.col_index[current_start_index + nnzs_max_row + ivc] = temp_current_col[ivc];
                }
                
                // check if row position needs to be changed
                // like before, only need to change if the swapping rows have difference nnzs
                if (nnzs_current != nnzs_max_row)
                {
                    int nnzs_diff = nnzs_max_row - nnzs_current;
                    A.row_position[i + 1] += nnzs_diff;
                }
            }
        }
        
        // Decomposition
        double s; // const divisor
        int found_start_index; //
        int found_next_start_index;
        int nnzs_found = 0;
        int nnzs_found_iright;
        int found_iright_start_index;
        int nnzs_current_iright = nnzs_max_row;
        int current_iright_start_index = current_start_index;
        int counter = 0;
        // from the next row, search for rows that also have values at the same col
        for (int k = A.row_position[i + 1]; k < A.nnzs; k++)
        {
            if (A.col_index[k] == i)
            {
                A.values[k] /= max_value;
                // from now on the lower part will become L except for the diagonal 1s
                // store the constant divisor
                s = A.values[k];
                for (int irp = 0; irp < A.rows; irp++)
                {
                    if ((A.row_position[irp] <= k) && (A.row_position[irp + 1] > k))
                    {
                        nnzs_found = A.row_position[irp + 1] - A.row_position[irp];
                        found_start_index = A.row_position[irp];
                        found_next_start_index = A.row_position[irp + 1];
                    }
                }

                found_iright_start_index = k + 1;
                nnzs_found_iright = found_next_start_index - found_iright_start_index;
                
                current_iright_start_index = current_start_index;
                nnzs_current_iright = nnzs_max_row;
                for (int ivc = current_start_index; ivc < current_start_index + nnzs_max_row; ivc++)
                {

                    if (A.col_index[ivc] <= i)
                    {
                        current_iright_start_index++;
                        nnzs_current_iright--;
                    }
                }
                
                // if the current row has 0 non-zeros, no change would be made
                if (nnzs_current_iright > 0)
                {
                    // find what will be the nnzs of found row after row manipulation
                    // as when there is a value at current row, found row needs to add a non-zero
                    int nnzs_found_iright_new = nnzs_current_iright + nnzs_found_iright;
                    for (int na = current_iright_start_index; na < current_iright_start_index + nnzs_current_iright; na++)
                    {
                        for (int nb = found_iright_start_index; nb < found_iright_start_index + nnzs_found_iright; nb++)
                        {
                            if (A.col_index[na] == A.col_index[nb]) nnzs_found_iright_new--;
                        }
                    }
                    
                    unique_ptr<double[]> temp_modified_value(new double[nnzs_found_iright_new]);
                    unique_ptr<int[]> temp_modified_col(new int[nnzs_found_iright_new]);
                    
                    // check if num_add is valid. Should be >= 0
                    int num_add = nnzs_found_iright_new - nnzs_found_iright;
                    if (num_add < 0)
                    {
                        cerr << "Invalid num_add." << endl;
                        return;
                    }
                    
                    vector<double> temp_value(0);
                    vector<int> temp_col(0);
                    
                    for (int c = current_iright_start_index; c < current_iright_start_index + nnzs_current_iright; c++)
                    {
                        int col_c = A.col_index[c];
                        bool flag = false;
                        int equal_index;
                        for (int f = found_iright_start_index; f < found_iright_start_index + nnzs_found_iright; f++)
                        {
                            if (A.col_index[f] == col_c)
                            {
                                equal_index = f;
                                flag = true;
                            }
                        }
                        if (flag)
                        {
                            temp_value.push_back(A.values[equal_index] - A.values[c] * s);
                            temp_col.push_back(A.col_index[equal_index]);
                        }
                        else
                        {
                            temp_value.push_back(0 - A.values[c] * s);
                            temp_col.push_back(A.col_index[c]);
                        }
                    }
                    
                    // get full record of found non-zeros at right of the row NO. i
                    // first get the overlapped points and calculate it (steps above)
                    // then for found row non-zeros that doean't have non-zerso at current row
                    // push them all and sort them according to their col_index
                    for (int f = found_iright_start_index; f < found_iright_start_index + nnzs_found_iright; f++)
                    {
                        if (!count(temp_col.begin(), temp_col.end(), A.col_index[f]))
                        {
                            temp_col.push_back(A.col_index[f]);
                            temp_value.push_back(A.values[f]);
                        }
                    }
                    
                    if (temp_col.size() != nnzs_found_iright_new)
                    {
                        cerr << "Invalid new length of found nnzs iright." << endl;
                        return;
                    }
                    
                    unique_ptr<int> tempc(new int);
                    unique_ptr<T> tempv(new T);
                    for (int ik = 1; ik < nnzs_found_iright_new; ik++)
                    {
                        for (int jk = 0; jk < nnzs_found_iright_new - 1; jk++)
                        {
                            if (temp_col[jk] > temp_col[ik])
                            {
                                swap(temp_col[jk], temp_col[ik]);
                                swap(temp_value[jk], temp_value[ik]);
                            }
                        }
                    }
                    
                    for (int check = 0; check < nnzs_found_iright - 1; check++)
                    {
                        if (temp_col[check] > temp_col[check + 1])
                        {
                            cerr << "Sorting failed." << endl;
                            return;
                        }
                    }

                    // if num_add == 0: no nnzs change, only change in values for iright at found row
                    if (num_add == 0)
                    {
                        for (int neg = 0; neg < nnzs_found_iright; neg++)
                        {
                            A.values[found_iright_start_index + neg] = temp_value[neg];
                            A.col_index[found_iright_start_index + neg] = temp_col[neg];
                        }
                    }
                    else
                    {
                        vector<double> new_value(A.nnzs + num_add);
                        vector<int> new_col(A.nnzs + num_add);

                        // assign unchanged values/col numbers before found iright
                        for (int ni = 0; ni < found_iright_start_index; ni++)
                        {
                            new_value[ni] = A.values[ni];
                            new_col[ni] = A.col_index[ni];
                        }

                        // insert temp modified values and col from the start of the found iright non-zeros
                        for (int temp_ni = 0; temp_ni < nnzs_found_iright_new; temp_ni++)
                        {
                            new_value[found_iright_start_index + temp_ni] = temp_value[temp_ni];
                            new_col[found_iright_start_index + temp_ni] = temp_col[temp_ni];
                        }

                        // push back the rest of non-zeros
                        for (int ni = found_iright_start_index + nnzs_found_iright; ni < A.nnzs; ni++)
                        {
                            new_value[ni + num_add] = A.values[ni];
                            new_col[ni + num_add] = A.col_index[ni];
                        }

                        for (int irp = 0; irp < A.rows; irp++)
                        {
                            if ((found_iright_start_index >= A.row_position[irp]) && (found_iright_start_index <= A.row_position[irp + 1]))
                            {
                                for (int irp_ = irp + 1; irp_ < A.rows + 1; irp_++)
                                {
                                    A.row_position[irp_] += num_add;
                                }
                            }
                        }
                        A.nnzs += num_add;
                        A.values.reset(new double[A.nnzs + num_add]);
                        A.col_index.reset(new int[A.nnzs + num_add]);

                        for (int pp = 0; pp < A.nnzs; pp++)
                        {
                            A.values[pp] = new_value[pp];
                            A.col_index[pp] = new_col[pp];
                        }
                    }
                }
                counter++;
                if (counter == A.rows)
                {
                    cerr << "Unable to decompose, please try another matrix" << endl;
                    return;
                }
            }
        }
        if (i == A.rows - 2) break;
    }
    
    // get P @ b
    auto* patb = new double[A.rows * 1];
    for (int i = 0; i < A.rows; i++)
    {
        patb[i] = b[p_col[i]];
    }
    auto* y = new double[A.rows * 1];
    // solve L(Ux) = P^-1 @ b
    A.forward_substitution(patb, y);
    A.backward_substitution(y, x);

    delete[] p_col;
    delete[] patb;
    delete[] y;
}

template<class T>
void gauss_elimination(Matrix<T>& A, T* x, T* b)
{
    // solves systems of linear eqs., with the Gaussian elimination method
    
    if (A.rows != A.cols)
    {
        // Assert that the no of eqs is equal to the no of uknowns
        cerr << "Cannot apply Gaussian Elimination on non-square matrix. \n";
        return;
    }

    // Loop constant initialization
    const int m = A.cols;
    int i, j, k;
    
    // Pivotisation / preconditioning
    // swap rows with ascending diagonal elements
    // from top to bottom of the matrix
    for (i = 0; i < m; i++) {
        for (k = i + 1; k < m; k++) {
            if (abs(A.values[i * m + i]) < abs(A.values[k * m + i])) {
                for (j = 0; j < m; j++) {
                    auto* temp = new double;
                    *temp = A.values[i * m + j];  // set temporary value before swaping
                    // Swap all row values given that the diagonal of
                    // row k is larger than the diagonal of row i
                    A.values[i * m + j] = A.values[k * m + j];
                    A.values[k * m + j] = *temp;
                    delete temp;
                }
                auto* temp2 = new double;
                *temp2 = b[i];
                // Similar for b
                *(b + i) = b[k];
                *(b + k) = *temp2;
                delete temp2;
            }
        }
    }

    // Perform Gauss-Elimination
    for (i = 0; i < m - 1; i++) {
        for (k = i + 1; k < m; k++) {
            auto* t = new double;
            *t = A.values[k * m + i] / A.values[i * m + i];
            for (j = 0; j < m; j++) {
                // set the coeff. equal to pivot elements or eliminate the variables
                A.values[k * m + j] = A.values[k * m + j] - *t * A.values[i * m + j];
            }
            *(b + k) = b[k] - *t * b[i];
            delete t;
        }
    }

    // Back-substitution
    for (i = m - 1; i >= 0; i--) {
        // Initialize x with the rhs of the last eq.
        *(x + i) = b[i];
        for (j = i + 1; j < m; j++) {
            // Subtracting all the lhs coefficient * x product from the x which is being calculated
            if (j != i) {
                *(x + i) = x[i] - A.values[i * m + j] * x[j];
            }
        }
        // Division of rhs by the coef of the x to be calculated
        *(x + i) = x[i] / A.values[i * m + i];
    }
}

template<class T>
void jacobi_dense(Matrix<T>& A, T* x, T* b, int maxit, double tolerance)
//user can input Matrix A and array b and can also define iteration tolerance and number of iteration times
{
    //set the condition that matrix A must be a square matrix
    if (A.rows != A.cols)
    {
        cerr << "Input matrix A must be a sqaure matrix!" << endl;
    }
    //have a guess of the size of output matrix, which is the same as the row's size of matrix A
    //use one new array to store the new solution, which avoids over-lapping the previous solution
    unique_ptr<T[]> x_new_array(new T[A.cols]);
    //initialize x array and new x array as zero arrays
    for (int i = 0; i < A.rows; i++)
    {
        x[i] = 0;
        x_new_array[i] = 0;
    }
    //mul_Ax is the dot product of A[i]*x[j] (except the diagonal values of A),
    //which is also the sum of ( A[i, :i] @ x[:i] ) and ( A[i, i+1:] @ x[i+1:] )
    unique_ptr<double[]> mul_Ax(new double[A.rows]);
    //total_sum is the dot product of two arrays( A[i,:] @ x[:] ), which will be used to calculate the norm
    unique_ptr<T[]> total_sum(new T[A.rows]);
    for (int k = 0; k <maxit; k++) //record the number of iteration
    {
        for (int i = 0; i < A.rows; i++)
        {
            total_sum[i] = 0;
            mul_Ax[i] = 0;
        }
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                if (i != j)
                {
                    //calculate the dot product of two arrays A[i]*x[j] (except the diagonal values of A) for each row
                    mul_Ax[i] += A.values[i * A.cols + j] * x[j];
                }
                x_new_array[i] = (1. / A.values[i * A.cols + i]) * (b[i] - mul_Ax[i]);
            }
        }
        //use pow_sum to calculate the norm
        double pow_sum = 0;
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                total_sum[i] += A.values[i * A.cols + j] * x_new_array[j];
            }
            pow_sum += pow(total_sum[i] - b[i], 2);
        }
        //calculate the norm difference between ( A @ x_new ) and b array
        double residual = sqrt(pow_sum);
        if (residual < tolerance) //if the residual is less than user-input tolerance, end the loop
        {
            break;
        }
        for (int i = 0; i < A.rows; i++)
        {
            //update the solution until the iteration over i finishes
            x[i] = x_new_array[i];
        }
    }
}

template<class T>
void jacobi_sparse(CSRMatrix<T>& A, T* x, T* b, int maxit, double tolerance)
//user can input Matrix A and array b and can also define iteration tolerance and number of iteration times
{
    //set the condition that matrix A must be a square matrix
    if (A.rows != A.cols)
    {
        cerr << "Input matrix A must be a sqaure matrix!" << endl;
    }
    //use one new array to store the new solution, which avoids over-lapping the previous solution
    unique_ptr<T[]> x_new_array(new T[A.cols]);
    //initialize x matrix and new x matrix as zero matrixs
    for (int i = 0; i < A.rows; i++)
    {
        x[i] = 0;
        x_new_array[i] = 0;
    }
    unique_ptr<int[]> row_diff(new int[A.cols]);
    //create a vector to store the diagonal values(A[i,i])
    vector<int> diag_values;
    //sum array pointer is to store the dot product of A[i]*x[j] (except the diagonal values of A),
    //which is also the sum of ( A[i, :i] @ x[:i] ) and ( A[i, i+1:] @ x[i+1:] )
    unique_ptr<T[]> sum(new T[A.rows]);
    //total_sum array pointer is to store the dot product of two arrays( A[i,:] @ x[:] ), which will be used to calculate the norm
    unique_ptr<T[]> total_sum(new double[A.rows]);
    //get the numbers of nonzero values for each row
    for (int i = 0; i < A.cols; i++)
    {
        row_diff[i] = A.row_position[i + 1] - A.row_position[i];
    }
    for (int k = 0; k < maxit; k++) //the number of iterations
    {
        for (int i = 0; i < A.rows; i++) //initialization of two dot pruduct array pointers
        {
            sum[i] = 0;
            total_sum[i] = 0;
        }
        int counter = 0;//use counter to track the index of each value
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < row_diff[i]; j++)
            {
                if (A.col_index[counter] == i) //find diagonal values(A[i,i])
                {
                    diag_values.push_back(A.values[counter]);
                }
                else //calculate the sum of ( A[i, :i] @ x[:i] ) and ( A[i, i+1:] @ x[i+1:] )
                {
                    sum[i] += A.values[counter] * x[A.col_index[counter]];
                }
                counter++;
            }
            x_new_array[i] = (1. / diag_values[i]) * (b[i] - sum[i]); //Jacobi stores the solution in one new array
        }
        //use pow_sum to calculate the norm
        double pow_sum = 0;
        for (int i = 0; i < A.rows; i++)
        {
            for (counter = A.row_position[i]; counter < A.row_position[i] + row_diff[i]; counter++)
            {
                total_sum[i] += A.values[counter] * x_new_array[A.col_index[counter]];
            }
            pow_sum += pow(total_sum[i] - b[i], 2);
        }
        //calculate the norm difference between ( A @ x_new ) and b array
        double residual = sqrt(pow_sum);
        if (residual < tolerance)
        {
            break;
        }
        for (int m = 0; m < A.rows; m++)
        {
            //update the solution until the iteration over i finishes
            x[m] = x_new_array[m];
        }
    }
}

template<class T>
void gauss_seidel_dense(Matrix<T>& A, T* x, T* b, int maxit, double er, double urf, int tiles)
{
    // solves systems of linear eqs., with the Gauss-Seidel method


    if (A.rows != A.cols)
    {
        // Assert that the no of eqs is equal to the no of uknowns
        cerr << "Cannot apply Gauss-Seidel on non-square matrix. \n";
        return;
    }

    // Initialization of looping constants
    const int n = A.cols;
    const int m = A.cols;
    int i, j, k, niter;
    double sum, xold;
    double rmx = 0.0;

    // initialisation of x
    for (i = 0; i < n; i++) {
        *(x + i) = 0.0;
    }

    // Pivotisation
    // swap rows with ascending diagonal elements
    // from top to bottom of the matrix
    for (i = 0; i < m; i++) {
        for (k = i + 1; k < m; k++) {
            if (abs(A.values[i * m + i]) < abs(A.values[k * m + i])) {
                for (j = 0; j < m; j++) {
                    auto* temp = new double;
                    *temp = A.values[i * m + j];  // set temporary value before swapin
                    // Swap all row values given that the diagonal of
                    // row k is larger than the diagonal of row i
                    A.values[i * m + j] = A.values[k * m + j];
                    A.values[k * m + j] = *temp;
                    delete temp;
                }
                auto* temp2 = new double;
                *temp2 = b[i];
                // Similar for b
                *(b + i) = b[k];
                *(b + k) = *temp2;
                delete temp2;
            }
        }
    }

    // Attempt to send coefficients for computation
    // that can fit in the cache memory (cache-aware).
    // Definition of loop start (minn)
    // loop finish (maxx) and tile size
    // (number of columns to be looped, tile_size)
    int* minn = new int[tiles + 1]{0};
    int tile_size = (int)(n /tiles);
    int* maxx = new int[tiles + 1]{0};
    for (int z = 0; z < tiles; z++){
        minn[z + 1] = z * tile_size;
        maxx[z + 1] = maxx[z] + tile_size;
    }
    // beginning of iterations
    for (niter = 0; niter < maxit; niter++) {

        double ea = 0.0;  // current error initialization
        
        // loop that goes through the tiles of A
        for (int z = 0; z < tiles; z++){
            // loop that calculates the coefficient within each tile
            for (i = minn[z + 1]; i < maxx[z + 1]; i++) {
                xold = x[i];
                sum = b[i];

                // Perform two loops to skip the diagonal elements
                // in the calculation of the sum
                for (j = 0; j <= i - 1; j++) {
                    sum += -A.values[i * n + j] * x[j];
                }

                for (j = i + 1; j < n; j++) {
                    sum += -A.values[i * n + j] * x[j];
                }

                *(x + i) = sum / A.values[i * n + i];

                // error and underelaxation
                double ern = abs(*(x + i) - xold);

                // keep the larger error between previous and current calculations
                ea = max(ea, ern);
                
                // Account for underelaxation factor to increase the stability
                // by considering the previous solution for x in the new x calculation.
                // The smaller the relaxation factor (urf), the more stable the method
                // becomes, however at a computational cost translated into an increase
                // of required iterations for convergence.
                *(x + i) = *(x + i) * urf + xold * (1. - urf);
            }
        }
        // new x's calculation


        // Checks for exit. If the current error (ea) is smaller
        // then the error input (er) the computations are terminated
        if (ea < er) {
            break;
        }

    }

    if (niter >= maxit) {
        // Warn the user in case maximum iterations have been reached
        // This could indicate the method has not succesfully converged
        // towards the proposed solution.
        cout << "Warning! Iterations' limit";
    }
}

template<class T>
void gauss_seidel_sparse(CSRMatrix<T>& A, T* x, T* b, int maxit, double tolerance)
{
    for (int i = 0; i < A.rows; i++) //initialize x array
    {
        x[i] = 0;
    }
    unique_ptr<int[]> row_diff(new int[A.cols]);
    vector<int> diag_values;
    //get the numbers of nonzero values for each row
    for (int i = 0; i < A.cols; i++)
    {
        row_diff[i] = A.row_position[i + 1] - A.row_position[i];
    }
    int m = A.rows;
    int n = A.cols;
    unique_ptr<T[]> sum(new T[m]);
    unique_ptr<T[]> total_sum(new double[A.rows]);
    for (int k = 0; k < maxit; k++) //the number of iterations
    {
        for (int i = 0; i < m; i++)
        {
            sum[i] = 0;
            total_sum[i] = 0;
        }
        int counter = 0;//use counter to track the index of each value
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < row_diff[i]; j++)
            {
                if (A.col_index[counter] == i) //find diagonal values(A[i,i])
                {
                    diag_values.push_back(A.values[counter]);
                }
                else
                {
                    sum[i] += A.values[counter] * x[A.col_index[counter]];//the result of ( A[i, :i] @ x[:i] ) + ( A[i, i+1:] @ x[i+1:] )
                    //which is actually a sum of dot products
                }
                counter++;
            }
            x[i] = 1. / diag_values[i] * (b[i] - sum[i]); //Gauss_seidle method is very similar to Jacobi method
            // The difference is that Gauss-Seidel method uses the latest updated values in the iteration while the
            //Jacobi method stores the values in one new array and uses the value obtained from the last i iteration
        }
        
        double pow_sum = 0;
        for (int i = 0; i < A.rows; i++)
        {
            for (counter = A.row_position[i]; counter < A.row_position[i] + row_diff[i]; counter++)
            {
                total_sum[i] += A.values[counter] * x[A.col_index[counter]];
            }
            pow_sum += pow(total_sum[i] - b[i], 2);
        }
        
        
        //calculate the error
        double residual = sqrt(pow_sum);
        if (residual < tolerance)
        {
            break;
        }
    }
}

template<class T>
void thomas(Matrix<T>& A, T* x, T* b)
{
    // solves systems of linear eqs., with the Gauss-Seidel method
    
    // Initialize 3*n array
    auto* A_array = new T[3 * A.rows];
    for (int i = 0; i < 3 * A.rows; i++)
    {
        A_array[i] = 0;
    }
    
    int i, j;
    // Perform Thomas method by only accessing
    // the tri-diagonal elements of A
    for (i = 0; i < A.rows; i++)
    {
        for (j = 0; j < A.cols; j++)
        {
            if (j == i + 1)
            {
                A_array[i+1] += A.values[i * A.cols + j];
            }
            if (j == i)
            {
                A_array[A.cols + j] += A.values[i * A.cols + j];
            }
            if (i == j + 1)
            {
                A_array[2 * A.cols + j] += A.values[i * A.cols + j];
            }
        }
    }

    T term;
    const int n = A.cols;
    A_array[2 * n] = A_array[2 * n] / A_array[1 * n];
    b[0] = b[0] / A_array[1 * n];
    
    //-- forward elimination for thomas
    for (i = 1; i < n; i++) {
        term = A_array[1 * n + i] - A_array[0 * n + i] * A_array[2 * n + i - 1];
        A_array[2 * n + i] = A_array[2 * n + i] / term;
        b[i] = (b[i] - A_array[0 * n + i] * b[i - 1]) / term;
    }
    //-- back substitution for thomas
    x[n - 1] = b[n - 1];
    for (i = n - 2; i >= 0; i--) {
            x[i] = b[i] - A_array[2 * n + i] * x[i + 1];
        }
    delete[] A_array;
}

template<class T>
void cholesky_fact(Matrix<T>& A, T* x, T* b) {
    
    // cholesky solver factorisation and substitutions component

    // Initialization of iteration counters
    int i, j, k;
    const int n = A.rows;
    
    // Initialization of upper triangular (au)
    // and lower triangular (al) A arrays.
    Matrix<T>* au = new Matrix<T>(n, n, true);
    Matrix<T>* al = new Matrix<T>(n, n, true);
    
    // Produce the elements of [L]^T
    for (i = 0; i < n - 1; i++) {   // !!!!!!!!!!!!
        for (j = i + 1; j < n; j++) {
            A.values[i * n + j] = A.values[j * n + i];
        }
    }

    // Copy the values of A in au and al
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            al->values[i * n + j] = 0.;
            au->values[i * n + j] = 0.;
            if (i >= j) {
                al->values[i * n + j] = A.values[i * n + j];
            }
            if (i <= j) {
                au->values[i * n + j] = A.values[i * n + j];
            }
        }
    }

    double* y = new double[n];
    *y = b[0] / al->values[0];
    
    // Perform forward and backward substitutions
    // to calculate final solution
    al->forward_sub(y, b);
    au->backward_sub(x, y);

    delete al;
    delete au;
    delete[] y;
}

template<class T>
void cholesky_dense(Matrix<T>& A, T* x, T* b) {
    
    // Cholesky solver routine for dense arrays

    // Initialization of iteration counters
    int i, j, k;
    const int n = A.rows;

    // Dense matrix calculations
    for (k = 0; k < n; k++) {

        A.values[k * n + k] = sqrt(A.values[k * n + k]);
        for (i = k + 1; i < n; i++) {
            A.values[i * n + k] = A.values[i * n + k] / A.values[k * n + k];
        }

        for (j = k + 1; j < n; j++) {
            for (i = j; i < n; i++) {
                A.values[i * n + j] -= A.values[i * n + k] * A.values[j * n + k];
            }
        }
    }
    
    // Call the factorization and substitutions component to produce a solution x
    cholesky_fact(A, x, b);
}


template<class T>
void cholesky_sparse(Matrix<T>& A, T* x, T* b) {
    
    // Cholesky solver routine for sparse arrays

    // Initialization of iteration counters
    int i, j, k;
    int n = A.rows;
    
    // Creation of a vector which will count
    // the number of non-zero elements in each column


    // Sparse matrix calculations
    for (k = 0; k < n; k++) {
        A.values[k * n + k] = sqrt(A.values[k * n + k]);
        vector<int> sk(n + 1);
        int nsk = 0;
        for (i = k + 1; i < n; i++) {
            A.values[i * n + k] = A.values[i * n + k] / A.values[k * n + k];

            // Count how many non-zero values there are in column k
            // and store that number in sk
            if (A.values[i * n + k] != 0.) {
                
                nsk += 1;
                sk[nsk] = i;
                
            }
        }

        // Perform calculations only if there was at least one non-zero values in
        // k column and only access these specific values.
        if (nsk > 0); {
            for (j = sk[1]; j <= sk[nsk]; j++) {
                for (i = j; i <= sk[nsk]; i++) {
                    A.values[i * n + j] = A.values[i * n + j] - A.values[i * n + k] * A.values[j * n + k];
                }
            }
        }
        
        // It must be noted however that while the above method works well for med-range dimensions
        // the performance gain decreases as the array dimensions increase. This happens due to the fact
        // that Cholesky produces an exponentially increasing amount of non-zero elements in positions that
        // used to be zero, as we move from the left to the right side of the coefficient array.
        // As a result, the strategy of avoiding non-zero elements becomes more ineffective for large arrays.
    }

    // Call the factorization and substitutions component to produce a solution x
    cholesky_fact(A, x, b);
}

template<class T>
void LU_dense_blas(Matrix<T>& A, T* x, T* b)
{
    if (A.rows != A.cols)
    {
        cerr << "Cannot decompose non-square matrix into LU. \n";
        return;
    }

    const int m = A.cols;
    auto* L = new Matrix<T>(m, m, true);
    auto* P_ = new Matrix<T>(m, m, true);
    
    for (int i = 0; i < m * m; i++)
    {
        L->values[i] = 0;
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (i == j) { P_->values[i * m + j] = 1; }
            else P_->values[i * m + j] = 0;
        }
    }

    for (int k = 0; k < m - 1; k++)
    {
        int index = k * A.cols + k;
        for (int i = k; i < A.cols; i++)
        {
            if (abs(A.values[i * A.cols + k]) > abs(A.values[index]))
            {
                index = i * A.cols + k;
            }
        }
        int j = index / A.cols;

        for (int i = 0; i < A.cols; i++)
        {
            T* numA = new T;
            *numA = A.values[k * A.rows + i];
            A.values[k * A.rows + i] = A.values[j * A.rows + i];
            A.values[j * A.rows + i] = *numA;

            T* numP_ = new T;
            *numP_ = P_->values[k * P_->rows + i];
            P_->values[k * P_->rows + i] = P_->values[j * P_->rows + i];
            P_->values[j * P_->rows + i] = *numP_;

            T* numL = new T;
            *numL = L->values[k * L->rows + i];
            L->values[k * L->rows + i] = L->values[j * L->rows + i];
            L->values[j * L->rows + i] = *numL;

            delete numA;
            delete numP_;
            delete numL;
        }
        
        int nvec = m - k;
        double* Avec_current = new double[nvec];
        for (int j = k; j < m; j++)
        {
            Avec_current[j - k] = A.values[k * m + j];
        }
        for (int i = k + 1; i < m; i++)
        {
            const double s = A.values[i * m + k] / A.values[k * m + k];
            L->values[i * m + k] = s;
            double* Avec_found = new double[m - k];
            for (int j = k; j < m; j++)
            {
                Avec_found[j - k] = A.values[i * m + j];
            }
            // vector operation using daxpy
            cblas_daxpy(m - k, -s, Avec_current, 1, Avec_found, 1);
            for (int p = k; p < m; p++)
            {
                A.values[i * m + p] = Avec_found[p - k];
            }
            delete[] Avec_found;
        }
        delete[] Avec_current;
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (i == j) { L->values[i * m + j] = 1; }
        }
    }

    auto* pinvb = new double[m * 1];
    P_->matVecMult(b, pinvb);
    auto* y = new double[m * 1];
    L->forward_sub(y, pinvb);
    A.backward_sub(x, y);

    delete L;
    delete P_;
    delete[] pinvb;
    delete[] y;
}

template<class T>
void gauss_seidel_dense_blas(Matrix<T>& A, T* x, T* b, int maxit, double er, double urf)
{
    if (A.rows != A.cols)
    {
        cerr << "Cannot apply Gauss-Seidel on non-square matrix. \n";
        return;
    }

    const int n = A.cols;
    const int m = A.cols;
    int i, j, k, niter;
    double sum, xold;
    double rmx = 0.0;

    for (i = 0; i < n; i++) {
        *(x + i) = 0.0;
    }

    for (i = 0; i < m; i++) {
        for (k = i + 1; k < m; k++) {
            if (abs(A.values[i * m + i]) < abs(A.values[k * m + i])) {
                for (j = 0; j < m; j++) {
                    auto* temp = new double;
                    *temp = A.values[i * m + j];
                    A.values[i * m + j] = A.values[k * m + j];
                    A.values[k * m + j] = *temp;
                    delete temp;
                }
                auto* temp2 = new double;
                *temp2 = b[i];
                *(b + i) = b[k];
                *(b + k) = *temp2;
                delete temp2;
            }
        }
    }
    
    for (niter = 0; niter < maxit; niter++) {

        double ea = 0.0;
        for (i = 0; i < n; i++) {
            xold = x[i];
            sum = b[i];
            auto* Avec = new T[A.cols];
            for (int j = 0; j < A.cols; j++)
            {
                Avec[j] = A.values[i * A.cols + j];
            }
            // dot product using blas
            sum -= cblas_ddot(A.rows, Avec, 1, x, 1);
            sum += A.values[i * A.cols + i] * x[i];
            delete[] Avec;

            *(x + i) = sum / A.values[i * n + i];
            
            double ern = abs(*(x + i) - xold);
            ea = max(ea, ern);
            *(x + i) = *(x + i) * urf + xold * (1. - urf);
        }
        if (ea < er) {
            break;
        }
    }

    if (niter >= maxit) {
        cout << "Warning! Iterations' limit";
    }
}

template<class T>
void jacobi_dense_blas(Matrix<T>& A, T* x, T* b, int maxit, double tolerance)
{
    if (A.rows != A.cols)
    {
        cerr << "Input matrix A must be a sqaure matrix!" << endl;
    }
    unique_ptr<T[]> x_new_array(new T[A.rows]);

    for (int i = 0; i < A.rows; i++)
    {
        x[i] = 0;
        x_new_array[i] = 0;
    }

    unique_ptr<T[]> total_sum(new T[A.rows]);
    unique_ptr<double[]> mul_Ax(new double[A.rows]);
    for (int k = 0; k <maxit; k++)
    {
        for (int i = 0; i < A.rows; i++)
        {
            total_sum[i] = 0;
            mul_Ax[i] = 0;
        }

        for (int i = 0; i < A.rows; i++)
        {
            auto* Avec = new T[A.cols];
            for (int j = 0; j < A.cols; j++)
            {
                Avec[j] = A.values[i * A.cols + j];
            }
            // cblas dot product function
            mul_Ax[i] += cblas_ddot(A.rows, Avec, 1, x, 1);
            mul_Ax[i] -= A.values[i * A.cols + i] * x[i];
            x_new_array[i] = (1. / A.values
                              [i * A.cols + i]) * (b[i] - mul_Ax[i]);
            delete[] Avec;
        }
        double pow_sum = 0;
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                total_sum[i] += A.values[i * A.cols + j] * x_new_array[j];

            }
            pow_sum += pow(total_sum[i] - b[i], 2);
        }
        double residual = sqrt(pow_sum);
        if (residual < tolerance)
        {
            break;
        }
        for (int i = 0; i < A.rows; i++)
        {
            x[i] = x_new_array[i];
        }
    }
}
