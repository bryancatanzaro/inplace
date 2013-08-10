#include "schedule.h"
#include "gcd.h"
#include "temporary.h"
#include "introspect.h"
#include "sm.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include <algorithm>
#include <typeinfo>
#include <iostream>


namespace inplace {
namespace detail {


template<typename T, typename F>
__global__ void smem_row_shuffle(int m, int n, T* d, F s);

template<typename SM, typename T, typename F, int WPT>
__global__ void register_row_shuffle(int m, int n, T* d, F s);

template<typename T, typename F>
__global__ void memory_row_shuffle(int m, int n, T* d, T* tmp, F s);



template<typename T, typename Schedule, typename SM>
struct shuffle_enactor {};

template<typename T, typename SM, int blks>
struct shuffle_enactor<T, smem<T, SM, blks>, SM> {
    bool enabled;
    static const int blk = smem<T, SM, blks>::blk;
    static const int lim = smem<T, SM, blks>::lim;
    shuffle_enactor(int n) {
        enabled = (n <= lim);
    }
    template<typename F>
    void operator()(T* data, int m, int n, F s, temporary_storage<T> temp) {
        int smem_bytes = sizeof(T) * n;
        smem_row_shuffle<<<m, blk, smem_bytes>>>(m, n, data, s);
    }
};

template<typename T, typename SM, int w, int b>
struct shuffle_enactor<T, reg<w, b>, SM> {
    bool enabled;
    static const int wpt = reg<w, b>::wpt;
    static const int blk = reg<w, b>::blk;
    shuffle_enactor(int n) {
        enabled = (n <= reg<w, b>::lim);
    }
    template<typename F>
    void operator()(T* data, int m, int n, F s, temporary_storage<T> temp) {
        register_row_shuffle<SM, T, F, wpt>
            <<<m, blk>>>(m, n, data, s);
    }
};


template<typename T, typename SM>
struct shuffle_enactor<T, memory, SM> {
    bool enabled;
    shuffle_enactor(int n) {
        enabled = true;
    }
    template<typename F>
    void operator()(T* data, int m, int n, F s, temporary_storage<T> temp) {
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, static_cast<T*>(temp), s);
    }
};

template<typename SM, typename T, typename F, typename Schedule, template<class, class, class> class Enactor>
struct enact_schedule {
    static void impl(T* data, int m, int n, F s, temporary_storage<T> temp) {
        Enactor<T, typename Schedule::head, SM>
            enactor(n);
        if (enactor.enabled) {
            enactor(data, m, n, s, temp);
        } else {
            enact_schedule<SM, T, F, typename Schedule::tail, Enactor>
                ::impl(data, m, n, s, temp);
        }
    }
};

template<typename SM, typename T, typename F, template<class, class, class> class Enactor>
struct enact_schedule<SM, T, F, memory, Enactor> {
    static void impl(T* data, int m, int n, F s, temporary_storage<T> temp) {
        Enactor<T, memory, SM> enactor(n);
        enactor(data, m, n, s, temp);
    }
};


template<typename T, typename F>
void shuffle_fn(T* data, int m, int n, F s, temporary_storage<T> temp) {
    int arch = current_sm();
    if (arch >= 305) {
        enact_schedule<sm_35, T, F, typename schedule<T, sm_35>::type, shuffle_enactor>
            ::impl(data, m, n, s, temp);
    } else if (arch >= 200) {
        enact_schedule<sm_20, T, F, typename schedule<T, sm_20>::type, shuffle_enactor>
            ::impl(data, m, n, s, temp);
    }
}

}

namespace c2r {

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n, T* tmp) {
    if (!row_major) {
        std::swap(m, n);
    }
    std::cout << "Doing C2R transpose of " << m << ", " << n << std::endl;
    temporary_storage<T> temp_storage(m, n, tmp);

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    if (c > 1) {
        detail::rotate(detail::c2r::prerotator(n/c), m, n, data);
    }
    detail::shuffle_fn(data, m, n, detail::c2r::shuffle(m, n, c, k), temp_storage);
    detail::rotate(detail::c2r::postrotator(m), m, n, data);
    int* temp_int = (int*)(static_cast<T*>(temp_storage));
    detail::scatter_permute(detail::c2r::scatter_postpermuter(m, n, c), m, n, data, temp_int);
}


void transpose(bool row_major, float* data, int m, int n, float* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}
void transpose(bool row_major, double* data, int m, int n, double* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}

}

namespace r2c {

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n, T* tmp) {
    if (row_major) {
        std::swap(m, n);
    }
    std::cout << "Doing R2C transpose of " << m << ", " << n << std::endl;
    temporary_storage<T> temp_storage(m, n, tmp);

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    int* temp_int = (int*)(static_cast<T*>(temp_storage));
    detail::scatter_permute(detail::r2c::scatter_prepermuter(m, n, c), m, n, data, temp_int);
    detail::rotate(detail::r2c::prerotator(m), m, n, data);
    detail::shuffle_fn(data, m, n, detail::r2c::shuffle(m, n, c, k), temp_storage);
    if (c > 1) {
        detail::rotate(detail::r2c::postrotator(n/c, m), m, n, data);
    }
}


void transpose(bool row_major, float* data, int m, int n, float* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}
void transpose(bool row_major, double* data, int m, int n, double* tmp) {
    transpose_fn(row_major, data, m, n, tmp);
}

}

}
