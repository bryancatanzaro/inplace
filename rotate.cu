#include "rotate.h"

#include <thrust/device_vector.h>
#include <iostream>

int main() {
    int m = 512;
    int n = 16000;
    thrust::device_vector<float> x(m * n);
    
    inplace::coarse_col_rotate<<<(n-1)/256+1, 256>>>(
        m, n, thrust::raw_pointer_cast(x.data()));
   
}
