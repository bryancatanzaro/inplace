#pragma once
#include "index.h"
namespace inplace {

__device__ int gcd(int a, int b) {
    int x = 0;
    int last_x = 1;
    int y = 1;
    int last_y = 0;
    while (b > 0) {
        int quotient = a / b;
        int new_b = a % b;
        a = b;
        b = new_b;
        int new_x = last_x - quotient * x;
        last_x = x;
        x = new_x;
        int new_y = last_y - quotient * y;
        last_y = y;
        y = new_y;
    }
    return a;
}

template<typename T>
__global__ void coarse_col_rotate(int m, int n, T* d) {

    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = global_index % m;

    int approximator = 32;
    int coarse_rotation = rotation_amount & (~(approximator-1));
    
    if ((global_index < n) && (coarse_rotation > 0)) {    
        row_major_index rm(m, n);
        int c = gcd(coarse_rotation, m);
        int inc = m - coarse_rotation;
        for(int b = 0; b < c; b++) {
            int pos = b;
            T prior = d[rm(pos, global_index)];
            int next_pos = pos + inc;
            if (next_pos >= m)
                next_pos -= m;
            while (next_pos >= c) {
                T temp = d[rm(next_pos, global_index)];
                pos = next_pos;
                d[rm(pos, global_index)] = prior;
                prior = temp;
                next_pos += inc;
                if (next_pos >= m)
                    next_pos -= m;
            }
            d[rm(next_pos, global_index)] = prior;

        }
    }
}

}
