#include <thrust/host_vector.h>
#include "gcd.h"
#include "index.h"
#include <algorithm>
#include <omp.h>
#include "reduced_math.h"

namespace inplace {
namespace openmp {
namespace detail {

struct prerotator {
    reduced_divisor m, b;
    prerotator() : m(1), b(1) {}
    prerotator(int _m, int _b) : m(_m), b(_b) {}
    int x;
    void set_j(const int& j) {
        x = b.div(j);
    }
    int operator()(const int& i) {
        return m.mod(i + x);
    }
};

struct postpermuter {
    reduced_divisor m;
    int n;
    reduced_divisor a;
    int j;
    postpermuter() : m(1), a(1) {}
    postpermuter(int _m, int _n, int _a) : m(_m), n(_n), a(_a) {}
    void set_j(const int& _j) {
        j = _j;
    }
    int operator()(const int& i) {
        return m.mod((i*n + j - a.div(i)));
    }
};

struct shuffle {
    int m, n, k;
    reduced_divisor b;
    reduced_divisor c;
    shuffle() : b(1), c(1) {}
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
                                              b(_n/_c), c(_c) {}
    int i;
    void set_i(const int& _i) {
        i = _i;
    }
    int f(const int& j) {
        int r = j + i * (n - 1);
        //The (int) casts here prevent unsigned promotion
        //and the subsequent underflow: c implicitly casts
        //int - unsigned int to
        //unsigned int - unsigned int
        //rather than to
        //int - int
        //Which leads to underflow if the result is negative.
        if (i - (int)c.mod(j) <= m - (int)c.get()) {
            return r;
        } else {
            return r + m;
        }
    }
    
    int operator()(const int& j) {
        int fij = f(j);
        unsigned int fijdivc, fijmodc;
        c.divmod(fij, fijdivc, fijmodc);
        //The extra mod in here prevents overflowing 32-bit int
        int term_1 = b.mod(k * b.mod(fijdivc));
        int term_2 = ((int)fijmodc) * (int)b.get();
        return term_1+term_2;
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
