#pragma once
#include "index.h"
#include "reduced_math.h"

namespace inplace {
namespace detail {

struct prerotator {
    int m, n;
//reduced_divisor m, n;
    int c, a;
    __host__ __device__ 
    prerotator(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    
    __host__ __device__ 
    void set_j(const int& j) {
        a = j * c / n;
    }
    
    __host__ __device__ 
    int operator()(const int& i) {
        return (i + a) % m;
    }
};

struct postpermuter {
    //reduced_divisor m;
    int m;
    int n, c, j;
    
    __host__ __device__ 
    postpermuter(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    
    __host__ __device__ 
    void set_j(const int& _j) {
        j = _j;
    }
    
    __host__ __device__ 
    int operator()(const int& i) {
        return ((i*n)-(i*c)/m+j) % m;
    }
};


struct shuffle {
    int m, n, k;
    reduced_divisor b;
    int c; //c can't be a reduced_divisor because it operates on long long
    __host__ __device__ 
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
                                              b(_n/_c), c(_c) {}
    int i;
    __host__ __device__ 
    void set_i(const int& _i) {
        i = _i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    __host__ __device__ 
    long long f(const int& j) {
        long long r = j + i *(n - 1);
        if ((i - (j % c)) <= (m - c)) {
            return r;
        } else {
            return r + m;
        }
    }
    
    __host__ __device__ 
    int operator()(const int& j) {
        long long fij = f(j);
        int term1 = (k *(fij/c)) % b;
        int term2 = (fij % c) * b;
        return term1 + term2;
    }
};


}
}

