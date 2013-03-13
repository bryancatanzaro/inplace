#pragma once

#include "introspect.h"
#include "array.h"
#include "index.h"

namespace inplace {
namespace detail {

struct prerotator {
    int m_m, m_n,m_c;
    __host__ __device__ __forceinline__
    prerotator(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    int m_a;
    __host__ __device__ __forceinline__
    void set_j(const int& j) {
        m_a = j * m_c / m_n;
    }
    __host__ __device__ __forceinline__
    int operator()(const int& i) {
        return (i + m_a) % m_m;
    }
};

struct postpermuter {
    int m_m, m_n, m_c, m_j;
    __host__ __device__ __forceinline__
    postpermuter(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    __host__ __device__ __forceinline__
    void set_j(const int& j) {
        m_j = j;
    }
    __host__ __device__ __forceinline__
    int operator()(const int& i) {
        return ((i*m_n)-(i*m_c)/m_m+m_j) % m_m;
    }
};


struct shuffle {
    int m_m, m_n, m_c, m_k;
    __host__ __device__ __forceinline__
    shuffle(int m, int n, int c, int k) : m_m(m), m_n(n), m_c(c), m_k(k) {}
    int m_i;
    __host__ __device__ __forceinline__
    void set_i(const int& i) {
        m_i = i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    __host__ __device__ __forceinline__
    long long f(const int& j) {
        long long r = j + m_i *(m_n - 1);
        if (m_i < (m_m + 1 - m_c + (j % m_c))) {
            return r;
        } else {
            return r + m_m;
        }
    }
    
    __host__ __device__ __forceinline__
    int operator()(const int& j) {
        long long fij = f(j);
        int term1 = (m_k *(fij/m_c)) % (m_n/m_c);
        int term2 = (fij % m_c) * (m_n/m_c);
        return (term1 + term2) % m_n;
    }
};


}
}

#include "register_ops.h"
#include "memory_ops.h"
