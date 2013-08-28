#include "util.h"
#include "skinny.h"
#include "save_array.h"




void test_transpose(int m, int n) {

    typedef int T;
    thrust::device_vector<T> x(m * n);
    thrust::device_vector<T> t(n);

    std::cout << "Checking results for transpose of a " << m << " x " <<
        n << " matrix" << std::endl;

    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(0) + m * n,
                      x.begin(),
                      inplace::tx_row_major_order<int>(n, m));
    inplace::save_array("golden.dat", thrust::raw_pointer_cast(x.data()), m, n);


    
    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(0) + m * n,
                      x.begin(),
                      inplace::row_major_order<int>(m, n));

    
    
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

        
    inplace::detail::c2r::skinny_transpose(
        thrust::raw_pointer_cast(x.data()),
        m, n,
        thrust::raw_pointer_cast(t.data()));


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "  Time: " << time << " ms" << std::endl;
    float gbs = (float)(2 * m * n * sizeof(T)) / (time * 1000000);
    std::cout << "  Throughput: " << gbs << " GB/s" << std::endl;
    
    inplace::save_array("output.dat", thrust::raw_pointer_cast(x.data()), m, n);
    

    
    
    bool correct = is_tx_row_major(x, m, n);
    if (correct) {
        std::cout << "PASSES" << std::endl << std::endl;
    } else {
        std::cout << "FAILS" << std::endl << std::endl;    
        exit(2);
    }

}

int main() {

    // for(int m = 1; m <= 32; m++) {
    //     for(int n = 1; n < 100; n++) {
    //         test_transpose(m, n);
    //     }
    // }

    test_transpose(31, 1e6);

}
