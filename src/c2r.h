#pragma once

#include "introspect.h"
#include "array.h"
#include "index.h"

namespace inplace {
namespace detail {

struct shuffle {
    int m, n, c, k;
    __host__ __device__ __forceinline__
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), c(_c), k(_k) {}
    int i;
    __host__ __device__ __forceinline__
    void set_i(const int& _i) {
        i = _i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    __host__ __device__ __forceinline__
    long long f(const int& j) {
        long long r = j + i *(n - 1);
        if (i < (m + 1 - c + (j % c))) {
            return r;
        } else {
            return r + m;
        }
    }
    
    __host__ __device__ __forceinline__
    int operator()(const int& j) {
        long long fij = f(j);
        int term1 = (k *(fij/c)) % (n/c);
        int term2 = (fij % c) * (n/c);
        return (term1 + term2) % n;
    }
};


}
}

#include "memory_ops.h"
#include "smem_ops.h"
