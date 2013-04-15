#include "rotate.h"

#include <thrust/device_vector.h>
#include <iostream>

template<typename T, typename Fn>
void print_array(const thrust::device_vector<T>& d, Fn index) {
    int m = index.m_m;
    int n = index.m_n;
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


int main() {
    int m = 512;
    int n = 64000;
    // int m = 32;
    // int n = 64;
    //int m = 33;
    //int n = 64;
    thrust::device_vector<int> x(m * n);
    thrust::copy(thrust::counting_iterator<int>(0),
                 thrust::counting_iterator<int>(0) + m * n,
                 x.begin());

    // print_array(x, inplace::row_major_index(m, n));
    std::cout << std::endl;

    //int block_size = 256;
    //int n_blocks = (n-1)/block_size + 1;

    // std::cout << "m: " << m << " n: " << n;
    // std::cout << " n_blocks: " << n_blocks << " block_size: " << block_size;
    // std::cout << std::endl;

    std::cout << "m: " << m << " n: " << n;
    std::cout << " n_blocks: " << (n-1)/32+1 << " block_size: 32x32";
    std::cout << std::endl;
 
    
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    

    inplace::fine_col_rotate<<<(n-1)/32+1, dim3(32,32)>>>(m, n, thrust::raw_pointer_cast(x.data()));
    // inplace::coarse_col_rotate<int, 4><<<n_blocks, block_size>>>(
    //     m, n, thrust::raw_pointer_cast(x.data()));
   
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(m * n * sizeof(float) * 2) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;


    //print_array(x, inplace::row_major_index(m, n));
    
}
