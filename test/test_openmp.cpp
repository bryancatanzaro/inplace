#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#define __forceinline__ 
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include "util/randint.h"
#include <iostream>
#include <algorithm>
#include "util.h"

#include "openmp.h"

#include <sys/time.h>

namespace inplace {
namespace openmp {

struct timer {
    struct timeval start, stop;
    void begin() {
        gettimeofday(&start, NULL);
    }
    float end() {
        gettimeofday(&stop, NULL);
        float result = (stop.tv_sec - start.tv_sec) * 1000.0f +
            (stop.tv_usec - start.tv_usec) * 1.0e-3f;
        return result;
    }
};
   


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

    timer the_timer;
    the_timer.begin();
    
    transpose(row_major,
              thrust::raw_pointer_cast(x.data()),
              m, n,
              thrust::raw_pointer_cast(t.data()));
    
    float time = the_timer.end();
    float gbs = (float)(2 * m * n * sizeof(double)) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;

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
    // for(int m = 1; m < 101; m++) {
    //     for(int n = 1; n < 101; n++) {
    //         inplace::openmp::test<double>(m, n, true);
    //         inplace::openmp::test<double>(m, n, false);
    //     }
    // }
    int min_dim = 1000;
    int max_dim = 10000;
    for(int i = 0; i < 1000; i++) {
        int m = inplace::detail::randint(min_dim, max_dim);
        int n = inplace::detail::randint(min_dim, max_dim);
        bool row_major = rand() & 2;
        inplace::openmp::test<double>(m, n, row_major);
    }
    return 0;
}

