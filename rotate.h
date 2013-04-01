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


template<typename T, int U>
__device__ __forceinline__
void unroll_rotate(T& prior, int& pos, int col, row_major_index rm, int inc, T* d) {
    T tmp[U];
    int positions[U];
    //Compute positions
    #pragma unroll
    for(int i = 0; i < U; i++) {
        positions[i] = pos;
        pos += inc;
        if (pos >= rm.m_m) pos -= rm.m_m;
    }
    //Load temporaries
    #pragma unroll
    for(int i = 0; i < U; i++) {
        tmp[i] = d[rm(positions[i], col)];
    }
    //Store results
    d[rm(positions[0], col)] = prior;
    #pragma unroll
    for(int i = 0; i < U-1; i++) {
        d[rm(positions[i+1], col)] = tmp[i];
    }
    prior = tmp[U-1];

}

template<typename T, int U>
__global__ void coarse_col_rotate(int m, int n, T* d) {
    int warp_id = threadIdx.x & 0x1f;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = (global_index - warp_id) % m;
    int col = global_index;
    
    if ((col < n) && (rotation_amount > 0)) {
        row_major_index rm(m, n);
        int c = gcd(rotation_amount, m);
        int l = m / c;
        int inc = m - rotation_amount;
        for(int b = 0; b < c; b++) {
            int pos = b;
            T prior = d[rm(pos, col)];
            pos += inc;
            if (pos >= m)
                pos -= m;
            int x = 0;
            for(; x < l - U + 1; x += U) {
                unroll_rotate<T, U>(prior, pos, col, rm, inc, d);
            }
            for(; x < l; x++) {
                unroll_rotate<T, 1>(prior, pos, col, rm, inc, d);
            }
            d[rm(pos, col)] = prior;

        }
    }
}

}
