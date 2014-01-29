#include "util.h"
#include "skinny.h"
#include "save_array.h"
#include "util/randint.h"
#include <cstdlib>



void test_transpose(int m, int n) {

    typedef double T;
    thrust::device_vector<T> x(m * n);

    std::cout << "Checking results for transpose of a " << m << " x " <<
        n << " matrix" << std::endl;

    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(0) + m * n,
                      x.begin(),
                      inplace::tx_row_major_order<int>(n, m));
    //inplace::save_array("golden.dat", thrust::raw_pointer_cast(x.data()), m, n);


    
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(0) + m * n,
                      x.begin(),
                      inplace::row_major_order<int>(m, n));

    
    
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (m < 32) {
        inplace::detail::c2r::skinny_transpose(
            thrust::raw_pointer_cast(x.data()),
            m, n);
    } else {
        inplace::detail::r2c::skinny_transpose(
            thrust::raw_pointer_cast(x.data()),
            n, m);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(2 * m * n * sizeof(T)) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;
    
    //inplace::save_array("output.dat", thrust::raw_pointer_cast(x.data()), m, n);
    

    
    
    bool correct = is_tx_row_major(x, m, n);
    if (correct) {
        std::cout << "PASSES" << std::endl << std::endl;
    } else {
        std::cout << "FAILS" << std::endl << std::endl;    
        exit(2);
    }

}


int main() {
    //Test R2C direction
    for(int i = 0; i < 10000; i++) {
        int n = inplace::detail::randint(2, 32);
        int m = inplace::detail::randint(10000, 10e6);
        test_transpose(m, n);
    }
    //Test C2R direction
    for(int i = 0; i < 10000; i++) {
        int m = inplace::detail::randint(2, 32);
        int n = inplace::detail::randint(10000, 10e6);
        test_transpose(m, n);
    }

}
