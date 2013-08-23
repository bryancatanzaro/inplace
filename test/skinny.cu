#include "util.h"
#include "skinny.h"



void test_transpose(int m, int n) {

    typedef int T;
    thrust::device_vector<T> x(m * n);
    thrust::device_vector<T> t(n);

    thrust::transform(thrust::counting_iterator<int>(0),
                      thrust::counting_iterator<int>(0) + m * n,
                      x.begin(),
                      inplace::tx_row_major_order<int>(m, n));

    print_array(x, inplace::row_major_index(m, n));

    
    inplace::detail::c2r::skinny_transpose(
        thrust::raw_pointer_cast(x.data()),
        m, n,
        thrust::raw_pointer_cast(t.data()));

    print_array(x, inplace::row_major_index(m, n));
    
    
}

int main() {

    test_transpose(5, 8);

}
