#include <mkl_trans.h>
#include <vector>

#include "../index.h"
#include <iostream>
using namespace inplace;

#include <cstdlib>

template<typename V, typename Fn>
void print_array(const V& d, Fn index) {
    int m = index.m_m;
    int n = index.m_n;
    typedef typename V::value_type T;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = d[index(i, j)];
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

void mkl_transpose(int m, int n) {
    std::cout << "Checking results for transpose of a " << m << " x " <<
        n << " matrix, in row major order";
    std::cout << std::endl;

    std::vector<double> array(m*n);
    for(int i = 0; i < m * n; i++) {
        array[i] = i;
    }
    // print_array(array, row_major_index(m, n));
    cudaEvent_t start,stop;
    float ms=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    mkl_dimatcopy('r', 't', m, n, 1.0, array.data(), n, m);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "  Time: " << ms << " ms" << std::endl;
    float bytes = (sizeof(double) * m * n * 2);
    float bandwidth = bytes / (ms * 1.0e6);

    std::cout << "  Throughput: " << bandwidth << " GB/s" << std::endl;

    
    // print_array(array, row_major_index(n, m));
    
}

void generate_random_size(int& m, int &n) {
    size_t e = 29440;
    while(true) {
        long long lm = rand() % e;
        long long ln = rand() % e;
        if (lm * ln > 0) {
            m = (int)lm;
            n = (int)ln;
            return;
        }
    }
}



int main() {
    for(int i = 0; i < 10; i++) {
        int m, n;
        generate_random_size(m, n);
        mkl_transpose(m, n);
    }

    
}
