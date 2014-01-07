#include <iostream>
#include "transpose.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <cstdlib>
#include "util.h"
#include <unistd.h>

using namespace inplace;


void visual_test(int m, int n) {
    thrust::device_vector<float> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), thrust::identity<int>());
    print_array(x, row_major_index(m, n));
    c2r::transpose(true, thrust::raw_pointer_cast(x.data()), m, n);
    std::cout << std::endl;
    //print_array(x, row_major_index(m, n));
    print_array(x, row_major_index(n, m));
}


template<typename T>
void time_test(int m, int n) {
    bool row_major = rand() & 2;

    std::cout << "Checking results for transpose of a " << m << " x " <<
        n << " matrix, in ";
    if (row_major) {
        std::cout << "row major order." << std::endl;
    } else {
        std::cout << "column major order." << std::endl;
    }
    
    thrust::device_vector<T> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), thrust::identity<T>());
    //Preallocate temporary storage.
    thrust::device_vector<T> t(max(m,n)*n_ctas());
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    //Simple heuristic
    if (m > n) {
    	inplace::c2r::transpose(row_major,
                                thrust::raw_pointer_cast(x.data()),
                                m, n);
    } else {
        inplace::r2c::transpose(row_major,
                               thrust::raw_pointer_cast(x.data()),
                               m, n);
    }


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(2 * m * n * sizeof(T)) / (time * 1000000);
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
        //exit(2);
    }
}

void generate_random_size(int& m, int &n) {
    int ub = 20000;
    int lb = 1000;
    int span = ub - lb;
    m = lb + rand() % span;
    n = lb + rand() % span;
}

int main() {
    // for(int m = 32; m < 1000; m++) {
    //     for(int n = 1; n < 1000; n++) {
    //         time_test<double>(m, n);
    //     }
    // }
    //visual_test(32, 6);
    // time_test<double>(32, 6);
    // time_test<double>(13985, 512);
    //time_test<double>(7200, 1800);
    for(int i = 0; i < 10000; i++) {
      int m, n;
      generate_random_size(m, n);
      time_test<double>(m, n);
    }
    //int n_pts = 501;
    //int l_bound = 1000;
    //int u_bound = 25000;
    //float delta = (float)(u_bound - l_bound) / (float)n_pts;
    //for(int m = l_bound; m < u_bound; m += (int)delta) {
    //    for(int n = l_bound; n < u_bound; n += (int)delta) {
    //        time_test<double>(m, n);
    //         //sleep(1);
    //    }
    //}
        
}
