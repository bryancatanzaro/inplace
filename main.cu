#include <iostream>
#include "transpose.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

template<typename T>
struct column_major_order {
    typedef T result_type;

    int m_m;
    int m_n;

    __host__ __device__
    column_major_order(const int& m, const int& n) :
        m_m(m), m_n(n) {}
    
    __host__ __device__ T operator()(const int& idx) {
        int row = idx % m_n;
        int col = idx / m_n;
        return row * m_m + col;
    }
};

template<typename T>
void print_array(int m, int n, const thrust::device_vector<T>& d) {
    thrust::host_vector<T> h = d;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = h[i * n + j];
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
    thrust::transform(c, c+(m*n), x.begin(), column_major_order<float>(m, n));
    print_array(m, n, x);
    inplace::transpose_rm(m, n, thrust::raw_pointer_cast(x.data()));
    std::cout << std::endl;
    print_array(n, m, x);

}

void time_test(int m, int n) {
    thrust::device_vector<float> x(m*n);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c+(m*n), x.begin(), column_major_order<float>(m, n));
    
   
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    inplace::transpose_rm(m, n, thrust::raw_pointer_cast(x.data()));


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(m * n * sizeof(float)) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl
              << std::endl;

    
}

int main() {
    for(int m = 1; m < 8; m++) {
        visual_test(m, 8);
        std::cout << "---------------------------------" << std::endl;
    }

    time_test(11520, 4896);

}
