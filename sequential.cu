#include "sequential.h"
#include "c2r.h"
#include "gcd.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
void sequential_col_op(int m, int n, T* d, T* tmp, F fn) {
    column_major_index cm(m, n);
    for(int j = 0; j < n; j++) {
        fn.set_j(j);
        for(int i = 0; i < m; i++) {
            int src_idx = fn(i);
            tmp[i] = d[cm(src_idx, j)];
        }
        for(int i = 0; i < m; i++) {
            d[cm(i, j)] = tmp[i];
        }
    }
}

template<typename T>
void sequential_row_shuffle(int m, int n, T* d, T* tmp, shuffle s) {
    column_major_index cm(m, n);
    for(int i = 0; i < m; i++) {
        s.set_i(i);
        for(int j = 0; j < n; j++) {
            tmp[j] = d[cm(i, s(j))];
        }
        for(int j = 0; j < n; j++) {
            d[cm(i, j)] = tmp[j];
        }
    }
}

template<typename T>
void sequential_transpose_fn(bool row_major, T* data, int m, int n, T* tmp) {
    if (!row_major) {
        int o = m;
        m = n;
        n = o;
    }

    
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        sequential_col_op(m, n, data, tmp, prerotator(m, n, c));
    }
    sequential_row_shuffle(m, n, data, tmp, shuffle(m, n, c, k));
    sequential_col_op(m, n, data, tmp, postpermuter(m, n, c));
}

}

void sequential_transpose(bool row_major, float* data, int m, int n, float* tmp) {
    detail::sequential_transpose_fn(row_major, data, m, n, tmp);
}

void sequential_transpose(bool row_major, double* data, int m, int n, double* tmp) {
    detail::sequential_transpose_fn(row_major, data, m, n, tmp);
}

}
