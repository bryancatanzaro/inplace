#include "c2r.h"
#include "schedule.h"

namespace inplace {
namespace detail {

// #if __CUDA_ARCH__ >= 350
// #define sm_type sm_35
// #elif __CUDA_ARCH__ >= 300
// #define sm_type sm_30
// #elif __CUDA_ARCH >= 200
// #define sm_type sm_20
// #else
// #define sm_type sm_10
// #endif


template<typename T, typename Schedule>
struct prerotate_enactor {
    T* data;
    int m, n;
    prerotator p;
    bool enabled;
    prerotate_enactor(T* _data, int _m, int _n, int _c, int _k)
        : data(_data), m(_m), n(_n), p(_m, _n, _c) {
        enabled = (_c > 1) && (m <= Schedule::lim);
    }
    void operator()() {
        inplace_col_op<T, prerotator, Schedule::wpt>
            <<<n, Schedule::blk>>>(m, n, data, p);
    }
};


template<typename T, typename Schedule>
struct shuffle_enactor {
    T* data;
    int m, n;
    shuffle s;
    bool enabled;
    shuffle_enactor(T* _data, int _m, int _n, int _c, int _k)
        : data(_data), m(_m), n(_n), s(_m, _n, _c, _k) {
        enabled = (n <= Schedule::lim);
    }
    void operator()() {
        inplace_row_shuffle<T, Schedule::wpt>
            <<<m, Schedule::blk>>>(m, n, data, s);
    }
};


template<typename T, typename Schedule>
struct postpermute_enactor {
    T* data;
    int m, n;
    postpermuter p;
    bool enabled;
    postpermute_enactor(T* _data, int _m, int _n, int _c, int _k)
        : data(_data), m(_m), n(_n), p(_m, _n, _c) {
        enabled = (m <= Schedule::lim);
    }
    void operator()() {
        inplace_col_op<T, postpermuter, Schedule::wpt>
            <<<n, Schedule::blk>>>(m, n, data, p);
    }
};

template<typename T, typename Schedule, template<class, class> class Enactor>
struct enact_schedule {
    static void impl(T* data, int m, int n, int c, int k) {
        Enactor<T, Schedule> enactor(data, m, n, c, k);
        if (enactor.enabled) {
            enactor();
        } else {
            enact_schedule<T, typename Schedule::tail, Enactor>
                ::impl(data, m, n, c, k);
        }
    }
};

template<typename T, template<class, class> class Enactor>
struct enact_schedule<T, nil, Enactor> {
    static void impl(T*, int, int, int, int) {}
};


template<typename T>
void prerotate_fn(T* data, int m, int n, int c, int k) {
    enact_schedule<T, typename schedule<T, sm_35>::type, prerotate_enactor>
        ::impl(data, m, n, c, k);
}

template<typename T>
void shuffle_fn(T* data, int m, int n, int c, int k) {
    enact_schedule<T, typename schedule<T, sm_35>::type, shuffle_enactor>
        ::impl(data, m, n, c, k);
}

template<typename T>
void postpermute_fn(T* data, int m, int n, int c, int k) {
    enact_schedule<T, typename schedule<T, sm_35>::type, postpermute_enactor>
        ::impl(data, m, n, c, k);
}


}

template<typename T>
void transpose(bool row_major, T* data, int m, int n) {
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
    detail::prerotate_fn(data, m, n, c, k);
    detail::shuffle_fn(data, m, n, c, k);
    detail::postpermute_fn(data, m, n, c, k);
}


//Explicit instantiation
template void transpose(bool, float*, int, int);
template void transpose(bool, double*, int, int);

}
