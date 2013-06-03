#include <thrust/host_vector.h>
#include "gcd.h"
#include "index.h"
#include "c2r.h"
#include <algorithm>

namespace inplace {
namespace sequential {
namespace detail {

template<typename T, typename F>
void col_shuffle(int m, int n, T* d, T* tmp, F fn) {
    row_major_index rm(m, n);
    for(int j = 0; j < n; j++) {
        fn.set_j(j);
        for(int i = 0; i < m; i++) {
            tmp[i] = d[rm(fn(i), j)];
        }
        for(int i = 0; i < m; i++) {
            d[rm(i, j)] = tmp[i];
        }
    }
}

template<typename T, typename F>
void row_shuffle(int m, int n, T* d, T* tmp, F fn) {
    row_major_index rm(m, n);
    for(int i = 0; i < m; i++) {
        fn.set_i(i);
        for(int j = 0; j < n; j++) {
            tmp[j] = d[rm(i, fn(j))];
        }
        for(int j = 0; j < n; j++) {
            d[rm(i, j)] = tmp[j];
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
        col_shuffle(m, n, data, tmp, inplace::detail::prerotator(m, n, c));
    }
    row_shuffle(m, n, data, tmp, inplace::detail::shuffle(m, n, c, k));
    col_shuffle(m, n, data, tmp, inplace::detail::postpermuter(m, n, c));
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
