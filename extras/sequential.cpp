#include "gcd.h"

namespace inplace {
namespace detail {

struct column_major_index {
    int m, n;

    column_major_index(const int& _m, const int& _n) :
        m(_m), n(_n) {}
    
    int operator()(const int& i, const int& j) const {
        return i + j * m;
    }
};

struct row_major_index {
    int m;
    int n;

    row_major_index(const int& _m, const int& _n) :
        m(_m), n(_n) {}

    int operator()(const int& i, const int& j) const {
        return j + i * n;
    }
};

struct prerotator {
    int m, n, c, a;
    
    prerotator(int m, int n, int c) : m(_m), n(_n), c(_c) {}
    
    void set_j(const int& j) {
        a = j * c / n;
    }
    
    int operator()(const int& i) {
        return (i + a) % m;
    }
};

struct postpermuter {
    int m, n, c, j;
    
    postpermuter(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    
    void set_j(const int& _j) {
        j = _j;
    }
    
    int operator()(const int& i) {
        return ((i*n)-(i*c)/m+j) % m;
    }
};

struct shuffle {
    int m, n, c, k, i;
    
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), c(_c), k(_k) {}
    
    void set_i(const int& _i) {
        i = _i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    
    long long f(const int& j) {
        long long r = j + i *(n - 1);
        if (i < (m + 1 - c + (j % c))) {
            return r;
        } else {
            return r + m;
        }
    }
    
    int operator()(const int& j) {
        long long fij = f(j);
        int term1 = (k *(fij/c)) % (n/c);
        int term2 = (fij % c) * (n/c);
        return (term1 + term2) % n;
    }
};

}

namespace sequential {
namespace detail {

template<typename T, typename F>
void col_shuffle(int m, int n, T* d, T* tmp, F fn) {
    row_major_index rm(m, n);
    for(int j = 0; j < n; j++) {
        fn.set_j(j);
        for(int i = 0; i < m; i++) {
            int src_idx = fn(i);
            tmp[i] = d[rm(src_idx, j)];
        }
        for(int i = 0; i < m; i++) {
            d[rm(i, j)] = tmp[i];
        }
    }
}

template<typename T>
void row_shuffle(int m, int n, T* d, T* tmp, shuffle s) {
    row_major_index rm(m, n);
    for(int i = 0; i < m; i++) {
        s.set_i(i);
        for(int j = 0; j < n; j++) {
            tmp[j] = d[rm(i, s(j))];
        }
        for(int j = 0; j < n; j++) {
            d[rm(i, j)] = tmp[j];
        }
    }
}

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n, T* tmp) {
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
        col_shuffle(m, n, data, tmp, prerotator(m, n, c));
    }
    row_shuffle(m, n, data, tmp, shuffle(m, n, c, k));
    col_shuffle(m, n, data, tmp, postpermuter(m, n, c));
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
