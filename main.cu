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
    thrust::device_vector<float> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), thrust::identity<int>());
    print_array(x, column_major_index(m, n));
    transpose(false, thrust::raw_pointer_cast(x.data()), m, n);
    std::cout << std::endl;
    print_array(x, column_major_index(n, m));
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

    
    transpose(row_major,
              thrust::raw_pointer_cast(x.data()),
              m, n);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(m * n * sizeof(T)) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;

    
    bool correct;
    if (row_major) {
        correct = is_ordered(x, tx_row_major_order<T>(n, m));
    } else {
        correct = is_ordered(x, tx_column_major_order<T>(n, m));
    }
    if (correct) {
        std::cout << "PASSES" << std::endl << std::endl;
    } else {
        std::cout << "FAILS" << std::endl << std::endl;
        exit(2);
    }
}

void generate_random_size(int& m, int &n) {
    size_t memory_size = gpu_memory_size();
    size_t ints_size = memory_size / sizeof(int);
    size_t e = 25600;//(size_t)sqrt(double(ints_size));
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
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    //visual_test(3,3);

    
    // for(int i = 1; i < 1000; i++) {
    //     for(int j = 1; j < 1000; j++) {
    //         time_test(i, j);
    //     }
    // }
            
    
    for(int i = 0; i < 1000; i++) {
        int m, n;
        generate_random_size(m, n);
        time_test<double>(m, n);
    }

    //time_test<float>(2047, 2046);
    
    //time_test(1045, 5735);
}
