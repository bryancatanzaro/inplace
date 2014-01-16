#include "util.h"
#include "skinny.h"
#include "save_array.h"
#include <cstdlib>



void test_transpose(int m, int n) {

    typedef double T;
    thrust::device_vector<T> x(m * n);
    int larger = max(m, n);
    
    thrust::device_vector<T> t(larger);

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
            m, n,
            thrust::raw_pointer_cast(t.data()));
    } else {
        inplace::detail::r2c::skinny_transpose(
            thrust::raw_pointer_cast(x.data()),
            n, m,
            thrust::raw_pointer_cast(t.data()));
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

int bounded_rand(int min, int max) {
    int range = max - min;
    return (rand() % range) + min;
}

int main() {
    //Test R2C direction
    for(int i = 0; i < 10000; i++) {
        int n = bounded_rand(2, 32);
        int m = bounded_rand(10000, 10e6);
        test_transpose(m, n);
    }
    //Test C2R direction
    for(int i = 0; i < 10000; i++) {
        int m = bounded_rand(2, 32);
        int n = bounded_rand(10000, 10e6);
        test_transpose(m, n);
    }

}
