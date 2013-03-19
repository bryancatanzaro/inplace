#include "c2r.h"
#include "schedule.h"
#include "gcd.h"
#include "temporary.h"
#include "introspect.h"
#include "sm.h"

namespace inplace {
namespace detail {

template<typename SM, typename T, typename F, int WPT>
__global__ void register_col_op(int, int, T*, F);

template<typename SM, typename T, int WPT>
__global__ void register_row_shuffle(int, int, T*, shuffle);


template<typename T, typename Schedule, typename SM>
struct prerotate_enactor {
    T* data;
    int m, n;
    prerotator p;
    bool enabled;
    temporary_storage<T> temp;
    prerotate_enactor(T* _data, int _m, int _n, int _c, int _k,
                      temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), p(_m, _n, _c), temp(_temp) {
        enabled = (_c > 1) && (m <= Schedule::lim);
    }
    void operator()() {
        register_col_op<SM, T, prerotator, Schedule::wpt>
            <<<n, Schedule::blk>>>(m, n, data, p);
    }
};

template<typename T, typename SM>
struct prerotate_enactor<T, memory, SM> {
    T* data;
    int m, n;
    prerotator p;
    bool enabled;
    temporary_storage<T> temp;
    prerotate_enactor(T* _data, int _m, int _n, int _c, int _k,
                      temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), p(_m, _n, _c), temp(_temp) {
        enabled = (_c > 1);
    }
    void operator()() {
        memory_col_op<T, prerotator>
            <<<n_ctas(), n_threads()>>>(m, n, data, static_cast<T*>(temp), p);
    }
};

template<typename T, typename Schedule, typename SM>
struct shuffle_enactor {
    T* data;
    int m, n;
    shuffle s;
    bool enabled;
    shuffle_enactor(T* _data, int _m, int _n, int _c, int _k,
                    temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), s(_m, _n, _c, _k) {
        enabled = (n <= Schedule::lim);
    }
    void operator()() {
        register_row_shuffle<SM, T, Schedule::wpt>
            <<<m, Schedule::blk>>>(m, n, data, s);
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

template<typename T, typename Schedule, typename SM>
struct postpermute_enactor {
    T* data;
    int m, n;
    postpermuter p;
    bool enabled;
    postpermute_enactor(T* _data, int _m, int _n, int _c, int _k,
                        temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), p(_m, _n, _c) {
        enabled = (m <= Schedule::lim);
    }
    void operator()() {
        register_col_op<SM, T, postpermuter, Schedule::wpt>
            <<<n, Schedule::blk>>>(m, n, data, p);
    }
};

template<typename T, typename SM>
struct postpermute_enactor<T, memory, SM> {
    T* data;
    int m, n;
    postpermuter p;
    bool enabled;
    temporary_storage<T> temp;
    postpermute_enactor(T* _data, int _m, int _n, int _c, int _k,
                    temporary_storage<T> _temp)
        : data(_data), m(_m), n(_n), p(_m, _n, _c), temp(_temp) {
        enabled = true;
    }
    void operator()() {
        memory_col_op<T>
            <<<n_ctas(), n_threads()>>>(m, n, data, static_cast<T*>(temp), p);
    }
};


template<typename SM, typename T, typename Schedule, template<class, class, class> class Enactor>
struct enact_schedule {
    static void impl(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
        Enactor<T, Schedule, SM> enactor(data, m, n, c, k, temp);
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
void prerotate_fn(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
    int arch = current_sm();
    if (arch >= 350) {
        enact_schedule<sm_35, T, typename schedule<T, sm_35>::type, prerotate_enactor>
            ::impl(data, m, n, c, k, temp);
    } else if (arch >= 200) {
        enact_schedule<sm_20, T, typename schedule<T, sm_20>::type, prerotate_enactor>
            ::impl(data, m, n, c, k, temp);
    }
}

template<typename T>
void shuffle_fn(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
    int arch = current_sm();
    if (arch >= 350) {
        enact_schedule<sm_35, T, typename schedule<T, sm_35>::type, shuffle_enactor>
            ::impl(data, m, n, c, k, temp);
    } else if (arch >= 200) {
        enact_schedule<sm_20, T, typename schedule<T, sm_20>::type, shuffle_enactor>
            ::impl(data, m, n, c, k, temp);
    }
}

template<typename T>
void postpermute_fn(T* data, int m, int n, int c, int k, temporary_storage<T> temp) {
    
    int arch = current_sm();
    if (arch >= 350) {
        enact_schedule<sm_35, T, typename schedule<T, sm_35>::type, postpermute_enactor>
            ::impl(data, m, n, c, k, temp);
    } else if (arch >= 200) {
        enact_schedule<sm_20, T, typename schedule<T, sm_20>::type, postpermute_enactor>
            ::impl(data, m, n, c, k, temp);
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
    temporary_storage<T> temp_storage(m, n, tmp);

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    detail::prerotate_fn(data, m, n, c, k, temp_storage);
    detail::shuffle_fn(data, m, n, c, k, temp_storage);
    detail::postpermute_fn(data, m, n, c, k, temp_storage);
}


void transpose(bool row_major, float* data, int m, int n, float* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}
void transpose(bool row_major, double* data, int m, int n, double* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}

}
