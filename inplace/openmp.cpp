#include <thrust/host_vector.h>
#include "gcd.h"
#include "index.h"
#include <algorithm>
#include <omp.h>

namespace inplace {
namespace openmp {
namespace detail {

struct prerotator {
    int m, b;
    prerotator() {}
    prerotator(int _m, int _b) : m(_m), b(_b) {}
    int x;
    void set_j(const int& j) {
        x = j / b;
    }
    int operator()(const int& i) {
        return (i + x) % m;
    }
};

struct postpermuter {
    int m, n, a, j;
    postpermuter() {}
    postpermuter(int _m, int _n, int _a) : m(_m), n(_n), a(_a) {}
    void set_j(const int& _j) {
        j = _j;
    }
    int operator()(const int& i) {
        return (i*n + j - (i/a)) % m;
    }
};


struct shuffle {
    int m, n, c, k, b;
    shuffle() {}
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), c(_c), k(_k) {
        b = n / c;
    }
    int i;
    void set_i(const int& _i) {
        i = _i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    long long f(const int& j) {
        long long r = j + i * (n - 1);
        if (i < (m + 1 - c + (j % c))) {
            return r;
        } else {
            return r + m;
        }
    }
    
    int operator()(const int& j) {
        long long fij = f(j);
        int term1 = (k *(fij/c)) % b;
        int term2 = (fij % c) * b;
        return term1 + term2;
    }
};



template<typename T, typename F>
void col_shuffle(int m, int n, T* d, T* tmp, F fn) {
    row_major_index rm(m, n);
    T* priv_tmp;
    F priv_fn;
    int tid;
    int i;
#pragma omp parallel private(tid, priv_tmp, priv_fn, i)
    {
        tid = omp_get_thread_num();
        priv_fn = fn;
        priv_tmp = tmp + m * tid;
#pragma omp for
        for(int j = 0; j < n; j++) {
            priv_fn.set_j(j);
            for(i = 0; i < m; i++) {
                priv_tmp[i] = d[rm(priv_fn(i), j)];
            }
            for(i = 0; i < m; i++) {
                d[rm(i, j)] = priv_tmp[i];
            }
        }
    }
}

template<typename T, typename F>
void row_shuffle(int m, int n, T* d, T* tmp, F fn) {
    row_major_index rm(m, n);
    T* priv_tmp;
    F priv_fn;
    int tid;
    int j;
#pragma omp parallel private(tid, priv_tmp, priv_fn, j)
    {
        tid = omp_get_thread_num();
        priv_fn = fn;
        priv_tmp = tmp + n * tid;
#pragma omp for
        for(int i = 0; i < m; i++) {
            priv_fn.set_i(i);
            for(j = 0; j < n; j++) {
                priv_tmp[j] = d[rm(i, priv_fn(j))];
            }
            for(j = 0; j < n; j++) {
                d[rm(i, j)] = priv_tmp[j];
            }
        }
    }
}

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n, T* tmp) {
    if (!row_major) {
        std::swap(m, n);
    }

    
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        col_shuffle(m, n, data, tmp,
                    inplace::openmp::detail::prerotator(m, n/c));
    }
    row_shuffle(m, n, data, tmp,
                inplace::openmp::detail::shuffle(m, n, c, k));
    col_shuffle(m, n, data, tmp,
                inplace::openmp::detail::postpermuter(m, n, m/c));
}

}

void transpose(bool row_major, float* data, int m, int n, float* tmp) {
    detail::transpose_fn(row_major, data, m, n, tmp);
}

void transpose(bool row_major, double* data, int m, int n, double* tmp) {
    detail::transpose_fn(row_major, data, m, n, tmp);
}

}
}
