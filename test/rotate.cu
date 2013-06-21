#include "rotate.h"

#include <thrust/device_vector.h>
#include <iostream>
#include <cassert>

template<typename T, typename Fn>
void print_array(const thrust::device_vector<T>& d, Fn index) {
    int m = index.m;
    int n = index.n;
    thrust::host_vector<T> h = d;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = h[index(i, j)];
            std::cout.width(5); std::cout << std::right;
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
}


struct fine_rotate_gold {
    typedef int result_type;
    int m, n;
    __host__ __device__ fine_rotate_gold(int _m, int _n) : m(_m), n(_n) {}
    __host__ __device__ int operator()(int idx) {
        int row = idx / n;
        int col = idx % n;
        int group_col = col & (~0x1f);
        int coarse_rotate = group_col % m;
        int col_rotate = col % m;
        int fine_rotate = col_rotate - coarse_rotate;
        if (fine_rotate < 0) fine_rotate += m;
        int src_row = row + fine_rotate;
        if (src_row >= m) src_row -= m;
        return (src_row * n) + col;
    }
};

struct overall_rotate_gold {
    typedef int result_type;
    int m, n;
    __host__ __device__ overall_rotate_gold(int _m, int _n) : m(_m), n(_n) {}
    __host__ __device__ int operator()(int idx) {
        int row = idx / n;
        int col = idx % n;
        int rotate = col % m;
        int src_row = row + rotate;
        if (src_row >= m) src_row -= m;
        return (src_row * n) + col;
    }
};

int main() {
    //int m = 64;
    //int n = 64;
    // int m = 32;
    // int n = 64;
    // int m = 33;
    // int n = 16;
    // for(int m = 32; m < 100; m++) {
    //     for(int n = 32; n < 100; n++) {
    int m = 66; int n = 33;
            typedef long long T;
            thrust::device_vector<T> x(m * n);
            thrust::copy(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(0) + m * n,
                         x.begin());

            print_array(x, inplace::row_major_index(m, n));
            std::cout << "m: " << m << " n: " << n << std::endl;

    
            cudaEvent_t start,stop;
            float time=0;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

    
            inplace::detail::postrotate(m, n, thrust::raw_pointer_cast(x.data()));
   
    
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
    
            std::cout << "  Time: " << time << " ms" << std::endl;
            float gbs = (float)(m * n * sizeof(T) * 2) / (time * 1000000);
            std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;
            std::cout << std::endl;
            // thrust::device_vector<int> y(m*n);
            // thrust::counting_iterator<int> c(0);
            // thrust::transform(c, c+m*n, y.begin(), fine_rotate_gold(m, n));
    
            print_array(x, inplace::row_major_index(m, n));
    
            assert(thrust::equal(x.begin(), x.end(), thrust::make_transform_iterator(
                                     thrust::counting_iterator<int>(0),
                                     overall_rotate_gold(m, n))));

    //     }
    // }
}
