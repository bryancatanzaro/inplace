#include <iostream>
#include "transpose.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <cstdlib>

using namespace inplace;

template<typename T, typename F>
bool is_ordered(const thrust::device_vector<T>& d,
                F fn) {
    return thrust::equal(d.begin(), d.end(),
                         thrust::make_transform_iterator(
                             thrust::counting_iterator<int>(0),
                             fn));
}

template<typename T, typename Fn>
void print_array(const thrust::device_vector<T>& d, Fn index) {
    int m = index.m_m;
    int n = index.m_n;
    thrust::host_vector<T> h = d;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = h[index(i, j)];
            if (x < 100) {
                std::cout << " ";
            }
            if (x < 10) {
                std::cout << " ";
            }
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
}

void visual_test(int m, int n) {
    thrust::device_vector<int> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), thrust::identity<int>());
    print_array(x, row_major_index(m, n));
    transpose_rm(m, n, thrust::raw_pointer_cast(x.data()));
    std::cout << std::endl;
    print_array(x, row_major_index(n, m));

}

void time_test(int m, int n) {
    std::cout << "Checking results for transpose of a " << m << " x " <<
        n << " matrix...";
    
    thrust::device_vector<int> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), thrust::identity<int>());
    //Preallocate temporary storage.
    thrust::device_vector<int> t(max(m,n)*n_ctas());
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    transpose_rm(m, n,
                 thrust::raw_pointer_cast(x.data()),
                 thrust::raw_pointer_cast(t.data()));


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(m * n * sizeof(float)) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl
              << std::endl;

    
    bool correct = is_ordered(x, tx_row_major_order<int>(n, m));
    if (correct) {
        std::cout << "PASSES" << std::endl;
    } else {
        std::cout << "FAILS" << std::endl;
        std::ostream_iterator<int> os(std::cout, " ");
        thrust::copy(x.begin(), x.end(), os);
        std::cout << std::endl;
        thrust::counting_iterator<int> c(0);
        thrust::copy(
            thrust::make_transform_iterator(c, tx_row_major_order<int>(n, m)),
            thrust::make_transform_iterator(c+m*n, tx_row_major_order<int>(n, m)),
            os);
        std::cout << std::endl;
        exit(2);
    }
}

void generate_random_size(int& m, int &n) {
    size_t memory_size = gpu_memory_size();
    size_t ints_size = memory_size / sizeof(int);
    size_t e = (size_t)sqrt(double(ints_size));
    while(true) {
        long long lm = rand() % e;
        long long ln = rand() % e;
        size_t extra = n_ctas() * max(lm, ln);
        if ((lm * ln > 0) && ((lm * (ln + extra)) < ints_size)) {
            m = (int)lm;
            n = (int)ln;
            return;
        }
    }
}

int main() {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    for(int i = 0; i < 1000; i++) {
        int m, n;
        generate_random_size(m, n);
        time_test(m, n);
    }

}
