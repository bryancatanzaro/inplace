#pragma once
#include "index.h"
#include "reduced_math.h"
#include "gcd.h"

namespace inplace {
namespace detail {

namespace c2r {

struct shuffle {
    int m, n, k;
    reduced_divisor b;
    reduced_divisor c;
    __host__
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k),
                                              b(_n/_c), c(_c) {}
    int i;
    __host__ __device__ 
    void set_i(const int& _i) {
        i = _i;
    }
    __host__ __device__
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
    
    __host__ __device__
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


struct prerotator {
    typedef int result_type;
    reduced_divisor b;
    __host__ prerotator(int _b) : b(_b) {}
    __host__ __device__
    int operator()(int j) const {
        return b.div(j);
    }
    __host__ __device__
    bool fine() const {
        return ((int)b.get() % 32) != 0;
    }
    __host__ __device__
    int master(int j, int w_id, int w) const {
        return j - w_id;
    }
};


struct postrotator {
    reduced_divisor m;
    __host__ postrotator(int _m) : m(_m) {}
    __host__ __device__
    int operator()(int j) const {
        return m.mod(j);
    }
    __host__ __device__
    bool fine() const {
        return true;
    }
    __host__ __device__
    int master(int j, int w_id, int w) const {
        return j - w_id;
    }
};

struct postpermuter {
    int m, n, a;
    __host__
    postpermuter(int _m, int _n, int _c) : m(_m), n(_n), a(_m/_c) {}
    __host__ __device__
    int operator()(int i) const {
        return (i * n - (i / a)) % m;
    }
    __host__ __device__
    int len() const {
        return m;
    }
};

}

namespace r2c {

struct prepermuter {
    int m; int n; int c; int a; int b; int q;
    __host__
    prepermuter(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {
        int d;
        extended_gcd(n/c, m/c, d, q);
        a = m / c;
        b = n / c;
    }
    __host__ __device__
    int operator()(int i) const {
        int k = ((c - 1) * i) % c;
        int l = ((c - 1 + i) / c);
        int r = k * a + ((l * q) % a);
        return r;
    }
    __host__ __device__
    int len() const {
        return m;
    } 
};

struct prerotator {
    typedef int result_type;
    reduced_divisor m;
    __host__ prerotator(int _m) : m(_m) {}
    __host__ __device__
    int operator()(int j) const {
        int jmodm = m.mod(j);
        return (jmodm == 0) ? 0 : (int) m.get() - jmodm;
    }
    __host__ __device__
    bool fine() const {
        return true;
    }
    __host__ __device__
    int master(int j, int w_id, int w) const {
        return j + w - 1 - w_id;
    }
};


struct shuffle {
    reduced_divisor m;
    reduced_divisor n;
    reduced_divisor b;
    __host__
    shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), b(_n/_c) {}
    int i;
    __host__ __device__ 
    void set_i(const int& _i) {
        i = _i;
    }    
    __host__ __device__
    int operator()(const int& j) {
        int r = m.mod(b.div(j) + i) + j * (int)m.get();
        return n.mod(r);
    }
};

struct postrotator {
    reduced_divisor b;
    typedef int result_type;
    int m;
    __host__ postrotator(int _b, int _m) : b(_b), m(_m) {}
    __host__ __device__
    int operator()(int j) const {
        int bdivj = (int)b.div(j);
        if (bdivj == 0) {
            return 0;
        } else {
            if (bdivj < m) {
                return m - bdivj;
            } else {
                return m + m - bdivj;
            }
        }
    }
    __host__ __device__
    bool fine() const {
        return ((int)b.get() % 32) != 0;
    }
    __host__ __device__
    int master(int j, int w_id, int w) const {
        return j + w - 1 - w_id;
    }
};

}

namespace c2r {

typedef r2c::prepermuter scatter_postpermuter;

}

namespace r2c {

typedef c2r::postpermuter scatter_prepermuter;

}


}
}
