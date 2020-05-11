#include <iostream>
#include <string>
#include <cblas.h> // if cblas installed
#include <memory>
#include <ctime>
#include "Matrix.h"
#include "CSRMatrix.h"
#include "solver.h"
#include "solver.cpp"
#include <algorithm>
#include <iomanip>
#include "testing.cpp"
#include "testing.h"

using namespace std;

int main()
{
    // elements of test_number represent
    // the switch for each test
    cout << "Tests: " << endl;
    cout << "--------------------" << endl;
    cout << "1. Solution validation: \nThis test will use a preset combination of Matrix and RHS to display solutions for all solvers in this library." << endl;
    cout << "2. Sparse vs Dense: \nCompares timings of different solvers." << endl;
    cout << "3. Dense vs BLAS: \nCompares different dense matrix solvers with their counterparts that have BLAS subroutines used." << endl;
    cout << "4. Container test: \nCompare the performance of different containers using Gaussian elimination algorithm." << endl;
    cout << "--------------------" << endl;
    cout << "Please type in the number of test (type 0 to review test options): ";
    while (true)
    {
        cout << "\nTest number: " << endl;
        int test_number;
        cin >> test_number;
        if (test_number == 1)
        {
            cout << "\n1. Solution validation: " << endl;
            char go_on;
            compare_results();
            cout << "\nContinue? (type y/n) " << endl;
            cin >> go_on;
            if (go_on == 'y') continue;
            else break;
        }
        else if (test_number == 2)
        {
            char print_char;
            bool print;
            cout << "Do you want to print solution values? (type y/n)" << endl;
            cin >> print_char;
            if (print_char == 'y') print = true;
            else print = false;
            
            cout << "What matrix type do you want to test? " << endl;
            int mat_int;
            cout << "1. dense\n" << "2. sparse\n" << "3. tridiagonal\n";
            cout << "Matrix type: ";
            cin >> mat_int;
            string mat_type;
            if (mat_int == 1) {mat_type = "dense";}
            else if (mat_int == 2) {mat_type = "sparse";}
            else if (mat_int == 3) {mat_type = "tridiagonal";}
            else
            {
                cout << "Unspecified matrix type." << endl;
                char go_on;
                cout << "\nContinue? (type y/n) " << endl;
                cin >> go_on;
                if (go_on == 'y') continue;
                else break;
            }
            
            double sparsity;
            if (mat_type == "sparse")
            {
                cout << "What's the sparsity: ";
                cin >> sparsity;
            }
            
            int n;
            cout << "How large is the matrix? " << endl;
            cin >> n;
            
            cout << "\n2. Sparse vs Dense: " << endl;
            time_dense_vs_sparse(n, mat_type, sparsity, print);
            char go_on;
            cout << "\nContinue? (type y/n) " << endl;
            cin >> go_on;
            if (go_on == 'y') continue;
            else break;
        }
        else if (test_number == 3)
        {
            char print_char;
            bool print;
            cout << "Do you want to print solution values? (type y/n)" << endl;
            cin >> print_char;
            if (print_char == 'y') print = true;
            else print = false;
            
            int n;
            cout << "How large is the matrix? " << endl;
            cin >> n;
            
            cout << "\n3. Dense vs Dense BLAS: " << endl;
            time_dense_vs_blas(n, print);
            
            char go_on;
            cout << "Continue? (type y/n) " << endl;
            cin >> go_on;
            if (go_on == 'y') continue;
            else break;
        }
        else if (test_number == 4)
        {
            cout << "\n4. Container test: " << endl;
            compare_containers();
            char go_on;
            cout << "Continue? (type y/n) " << endl;
            cin >> go_on;
            if (go_on == 'y') continue;
            else break;
        }
        else
        {
            cout << "Can't recogonise test number. \nProcess aborted.\n";
            break;
        }
        
        cout << "--------------------" << endl;
        cout << "1. Solution validation" << endl;
        cout << "2. Sparse vs Dense" << endl;
        cout << "3. Dense vs BLAS" << endl;
        cout << "4. Container test" << endl;
        cout << "Please type in the number of test (type 0 to review test options): ";
        cout << "--------------------" << endl;
    }
    return 0;
}
