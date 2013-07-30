#pragma once
#include "index.h"
#include "reduced_math.h"

namespace inplace {
namespace detail {

struct prerotator {
    reduced_divisor_32 m, b;
    int c, a;
    __host__  
    prerotator(int _m, int _n, int _c) : m(_m), b(_n / _c), c(_c) {}
    
    __host__ __device__ 
    void set_j(const int& j) {
        a = b.div(j);
    }
    
    __host__ __device__ 
    int operator()(const int& i) {
        return m.mod(i + a);
    }
};

struct postpermuter {
    reduced_divisor_32 m;
    int n, c, j;
    
    __host__ 
    postpermuter(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    
    __host__ __device__ 
    void set_j(const int& _j) {
        j = _j;
    }
    
    __host__ __device__ 
    int operator()(const int& i) {
        return m.mod((i*n)-m.div(i*c)+j);
    }
};


struct shuffle {
    int m, n, k;
    reduced_divisor_32 b;
    reduced_divisor_32 c;
    __host__
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
                                              b(_n/_c), c(_c) {}
    int i;
    __host__ __device__ 
    void set_i(const int& _i) {
        i = _i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    // __host__ __device__ 
    // long long f(const int& j) {
    //     long long r = j + (long long)i * (long long)(n - 1);
    //     if ((i - (int)c.mod(j)) <= (m - (int)c.get())) {
    //         return r;
    //     } else {
    //         return r + m;
    //     }
    // }
    __host__ __device__
    int f(const int& j) {
        int r = j + i * (n - 1);
        if (i - (int)c.mod(j) <= m - (int)c.get()) {
            return r;
        } else {
            return r + m;
        }
    }
    
    // __host__ __device__ 
    // int operator()(const int& j) {
    //     long long fij = f(j);
    //     unsigned long long fijdivc;
    //     unsigned long long fijmodc;
    //     c_64.divmod(fij, fijdivc, fijmodc);
        
    //     int term1 = b.mod((k *((int)fijdivc)));
    //     int term2 = ((int)fijmodc) * (int)b.get();
    //     return term1 + term2;
    // }
    __host__ __device__
    int operator()(const int& j) {
        int fij = f(j);
        // int fijdivc = fij / c.get();
        // int fijmodc = fij % c.get();//fij - (fijdivc * c.get());
        unsigned int fijdivc, fijmodc;
        c.divmod(fij, fijdivc, fijmodc);
        int term_1 = b.mod(k * b.mod(fijdivc));
        int term_2 = ((int)fijmodc) * (int)b.get();
        return term_1+term_2;
    }
};


}
}

