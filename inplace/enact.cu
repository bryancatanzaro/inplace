#include "c2r.h"
#include "schedule.h"
#include "gcd.h"
#include "temporary.h"
#include "introspect.h"
#include "sm.h"
#include "rotate.h"
#include "permute.h"
#include "smem_ops.h"
#include "register_ops.h"
#include "memory_ops.h"
#include <algorithm>
#include <typeinfo>

namespace inplace {
namespace detail {

template<typename SM, typename T, int WPT>
__global__ void register_row_shuffle(int, int, T*, shuffle);


template<typename T, typename Schedule, typename SM>
struct shuffle_enactor {};

template<typename T, typename SM, int blks>
struct shuffle_enactor<T, smem<T, SM, blks>, SM> {
    T* data;
    int m, n;
    shuffle s;
    bool enabled;
    static const int blk = smem<T, SM, blks>::blk;
    static const int lim = smem<T, SM, blks>::lim;
    shuffle_enactor(T* _data, int _m, int _n, int _c, int _k,
                    temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), s(_m, _n, _c, _k) {
        enabled = (n <= lim);
    }
    void operator()() {
        int smem_bytes = sizeof(T) * n;
        smem_row_shuffle<<<m, blk, smem_bytes>>>(m, n, data, s);
    }
};

template<typename T, typename SM, int w, int b>
struct shuffle_enactor<T, reg<w, b>, SM> {
    T* data;
    int m, n;
    shuffle s;
    bool enabled;
    static const int wpt = reg<w, b>::wpt;
    static const int blk = reg<w, b>::blk;
    shuffle_enactor(T* _data, int _m, int _n, int _c, int _k,
                    temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), s(_m, _n, _c, _k) {
        enabled = (n <= reg<w, b>::lim);
    }
    void operator()() {
        register_row_shuffle<SM, T, wpt>
            <<<m, blk>>>(m, n, data, s);
    }
};


template<typename T, typename SM>
struct shuffle_enactor<T, memory, SM> {
    T* data;
    int m, n;
    shuffle s;
    bool enabled;
    temporary_storage<T> temp;
    shuffle_enactor(T* _data, int _m, int _n, int _c, int _k,
                    temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), s(_m, _n, _c, _k), temp(_temp) {
        enabled = true;
    }
    void operator()() {
        memory_row_shuffle<T>
            <<<n_ctas(), n_threads()>>>(m, n, data, static_cast<T*>(temp), s);
    }
};

template<typename SM, typename T, typename Schedule, template<class, class, class> class Enactor>
struct enact_schedule {
    static void impl(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
        Enactor<T, typename Schedule::head, SM>
            enactor(data, m, n, c, k, temp);
        if (enactor.enabled) {
            enactor();
        } else {
            enact_schedule<SM, T, typename Schedule::tail, Enactor>
                ::impl(data, m, n, c, k, temp);
        }
    }
};

template<typename SM, typename T, template<class, class, class> class Enactor>
struct enact_schedule<SM, T, memory, Enactor> {
    static void impl(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
        Enactor<T, memory, SM> enactor(data, m, n, c, k, temp);
        if (enactor.enabled) {
            enactor();
        }
    }
};


template<typename T>
void shuffle_fn(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
    int arch = current_sm();
    if (arch >= 305) {
        enact_schedule<sm_35, T, typename schedule<T, sm_35>::type, shuffle_enactor>
            ::impl(data, m, n, c, k, temp);
    } else if (arch >= 200) {
        enact_schedule<sm_20, T, typename schedule<T, sm_20>::type, shuffle_enactor>
            ::impl(data, m, n, c, k, temp);
    }
}

}

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n, T* tmp) {
    if (!row_major) {
        std::swap(m, n);
    }
    std::cout << "Doing transpose of " << m << ", " << n << std::endl;
    temporary_storage<T> temp_storage(m, n, tmp);

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        detail::prerotate(c, m, n, data);
    }
    detail::shuffle_fn(data, m, n, c, k, temp_storage);
    detail::postrotate(m, n, data);
    int* temp_int = (int*)(static_cast<T*>(temp_storage));
    detail::postpermute(m, n, c, data, temp_int);
}


void transpose(bool row_major, float* data, int m, int n, float* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}
void transpose(bool row_major, double* data, int m, int n, double* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}

}
