#pragma once
#include "index.h"
namespace inplace {

__device__ __forceinline__
unsigned int ctz(unsigned int x) {
    return __ffs(x) - 1;
}

__device__ __forceinline__
unsigned int gcd(unsigned int x, unsigned int y) {
    if (x == 0) return y;
    if (y == 0) return x;
    unsigned int cf2 = ctz(x | y);
    x >>= ctz(x);
    while (true) {
        y >>= ctz(y);
        if (x == y) break;
        if (x > y) {
            unsigned int t = x; x = y; y = t;
        }
        if (x == 1) break;
        y -= x;
    }
    return x << cf2;
}

__device__ __forceinline__
unsigned int div_up(unsigned int a, unsigned int b) {
    return (a-1)/b + 1;
}

__device__ __forceinline__
unsigned int div_down(unsigned int a, unsigned int b) {
    return a / b;
}

template<typename T>
__global__ void coarse_col_rotate(int m, int n, T* d) {
    int warp_id = threadIdx.x & 0x1f;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = (global_index - warp_id) % m;
    int col = global_index;
    
    if ((col < n) && (rotation_amount > 0)) {
        row_major_index rm(m, n);
        int c = gcd(rotation_amount, m);
        int inc = m - rotation_amount;
        for(int b = 0; b < c; b++) {
            int pos = b;
            T prior = d[rm(pos, col)];
            int next_pos = pos + inc;
            if (next_pos >= m)
                next_pos -= m;
            while (next_pos >= c) {
                T temp = d[rm(next_pos, col)];
                pos = next_pos;
                d[rm(pos, col)] = prior;
                prior = temp;
                next_pos = pos + inc;
                if (next_pos >= m)
                    next_pos -= m;
            }
            d[rm(next_pos, col)] = prior;

        }
    }
}

}
