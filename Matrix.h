#pragma once    
#include <vector>
#include <tuple>
#include <memory>

using namespace std;

template<class T>
class Matrix
{
public:
	/////// public methods
	
	// constructor where we want to preallocate memory: own our own memory
	Matrix(int rows, int cols, bool preallocate);
	
	// constructor where we want to preallocate memoryalready preallocated memory outside
	// dont own our own memory
	Matrix(int rows, int cols, T* values_ptr);

    // hard copy constructor
    // avoid changing the copy of the matrix due to the change of the original one
    Matrix(const Matrix &B);

	// destructor
	// virtual ~Matrix() = 0; pure virtual function: the sub class MUST overwrite this func
	virtual ~Matrix(); //the sub class CAN overwrite this func

	// print matrix values
	void printValues();
	virtual void printMatrix();

	// some basic functions
	void transpose(Matrix<T>& itself);
    // generate random n x n dense matrix
    //if dom, dominant on the diagonal
    void genRanDense(bool dom);
    // generate random n x n sparse matrix, sparsity(<1) represents the sparsity of the sparse matrix
    void genRanSparse(double sparsity, bool dom);
    //generate random tri matrix for testing Thomas solution
    void genRanTri(bool dom);
	void matMatMult(Matrix& mat_right, Matrix& output);
	void matVecMult(T* vec, T* output);
    // forward substitution, assume this is a lower triangle
    void forward_sub(T* x, T* b);
    // backward substitution, assume this is a upper triangle
    void backward_sub(T* x, T* b);
	
	/////// public variables
	// matrix size;
	// explicitly using the c++11 nullptr;
	unique_ptr<T[]> values;
	int rows = -1;
	int cols = -1;

	// private variables: no need for other classes to know
	// values
protected:
	bool preallocated = false;
private:
	int size_of_values = -1;
	

};
