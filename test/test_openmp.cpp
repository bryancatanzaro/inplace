#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <cstdlib>
#include <algorithm>
#include "util.h"

#include "openmp.h"

namespace inplace {
namespace openmp {

template<typename T>
void test(int m, int n, bool row_major=true) {
    std::cout << "Checking results for transpose of a " << m << " x " <<
        n << " matrix, in ";
    if (row_major) {
        std::cout << "row major order." << std::endl;
    } else {
        std::cout << "column major order." << std::endl;
    }
    
    thrust::host_vector<T> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), thrust::identity<T>());
    //Preallocate temporary storage.
    int max_threads = omp_get_max_threads();
    thrust::host_vector<T> t(std::max(m,n) * max_threads);
    
    
    transpose(row_major,
              thrust::raw_pointer_cast(x.data()),
              m, n,
              thrust::raw_pointer_cast(t.data()));


    bool correct;
    if (row_major) {
        correct = is_tx_row_major(x, m, n);
    } else {
        correct = is_tx_col_major(x, m, n);
    }
    if (correct) {
        std::cout << "PASSES" << std::endl << std::endl;
    } else {
        std::cout << "FAILS" << std::endl << std::endl;
        exit(2);
    }
}

}
}

int main() {
    for(int m = 1; m < 101; m++) {
        for(int n = 1; n < 101; n++) {
            inplace::openmp::test<double>(m, n, true);
            inplace::openmp::test<double>(m, n, false);
        }
    }

    int max_dim = 10000;    
    for(int i = 0; i < 1000; i++) {
        int m = (rand() & (max_dim - 1)) + 1;
        int n = (rand() & (max_dim - 1)) + 1;
        bool row_major = rand() & 2;
        inplace::openmp::test<double>(m, n, row_major);
    }
    return 0;
}

