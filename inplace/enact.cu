#include "schedule.h"
#include "gcd.h"
#include "introspect.h"
#include "sm.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include "skinny.h"
#include <algorithm>


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
    void operator()(T* data, int m, int n, F s) {
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
    void operator()(T* data, int m, int n, F s) {
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
    void operator()(T* data, int m, int n, F s) {
        T* temp;
        cudaMalloc(&temp, sizeof(T) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
    }
};

template<typename SM, typename T, typename F, typename Schedule, template<class, class, class> class Enactor>
struct enact_schedule {
    static void impl(T* data, int m, int n, F s) {
        Enactor<T, typename Schedule::head, SM>
            enactor(n);
        if (enactor.enabled) {
            enactor(data, m, n, s);
        } else {
            enact_schedule<SM, T, F, typename Schedule::tail, Enactor>
                ::impl(data, m, n, s);
        }
    }
};

template<typename SM, typename T, typename F, template<class, class, class> class Enactor>
struct enact_schedule<SM, T, F, memory, Enactor> {
    static void impl(T* data, int m, int n, F s) {
        Enactor<T, memory, SM> enactor(n);
        enactor(data, m, n, s);
    }
};


template<typename T, typename F>
void shuffle_fn(T* data, int m, int n, F s) {
    int arch = current_sm();
    if (arch >= 305) {
        enact_schedule<sm_35, T, F, typename schedule<T, sm_35>::type, shuffle_enactor>
            ::impl(data, m, n, s);
    } else if (arch >= 200) {
        enact_schedule<sm_20, T, F, typename schedule<T, sm_20>::type, shuffle_enactor>
            ::impl(data, m, n, s);
    }
}

}

namespace c2r {

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n) {
    if (!row_major) {
        std::swap(m, n);
    }
    //std::cout << "Doing C2R transpose of " << m << ", " << n << std::endl;

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
    detail::shuffle_fn(data, m, n, detail::c2r::shuffle(m, n, c, k));
    detail::rotate(detail::c2r::postrotator(m), m, n, data);
    int* temp_int;
    cudaMalloc(&temp_int, sizeof(int) * m);
    detail::scatter_permute(detail::c2r::scatter_postpermuter(m, n, c), m, n, data, temp_int);
    cudaFree(temp_int);
}


void transpose(bool row_major, float* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}
void transpose(bool row_major, double* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}

}

namespace r2c {

template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n) {
    if (row_major) {
        std::swap(m, n);
    }
    //std::cout << "Doing R2C transpose of " << m << ", " << n << std::endl;

    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }
    int* temp_int;
    cudaMalloc(&temp_int, sizeof(int) * m);
    detail::scatter_permute(detail::r2c::scatter_prepermuter(m, n, c), m, n, data, temp_int);
    cudaFree(temp_int);
    detail::rotate(detail::r2c::prerotator(m), m, n, data);
    detail::shuffle_fn(data, m, n, detail::r2c::shuffle(m, n, c, k));
    if (c > 1) {
        detail::rotate(detail::r2c::postrotator(n/c, m), m, n, data);
    }
}


void transpose(bool row_major, float* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}
void transpose(bool row_major, double* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}

}


template<typename T>
void transpose_fn(bool row_major, T* data, int m, int n) {
    bool small_m = m < 32;
    bool small_n = n < 32;
    if (!small_m && small_n) {
        std::swap(m, n);
        if (!row_major) {
            inplace::detail::c2r::skinny_transpose(
                data, m, n);
        } else {
            inplace::detail::r2c::skinny_transpose(
                data, m, n);
        }
    } else if (small_m) {
        if (!row_major) {
            inplace::detail::r2c::skinny_transpose(
                data, m, n);
        } else {
            inplace::detail::c2r::skinny_transpose(
                data, m, n);
        }
    } else {
        bool m_greater = m > n;
        if (m_greater ^ row_major) {
            inplace::r2c::transpose(row_major, data, m, n);
        } else {
            inplace::c2r::transpose(row_major, data, m, n);
        }
    }
}

void transpose(bool row_major, float* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}
void transpose(bool row_major, double* data, int m, int n) {
    transpose_fn(row_major, data, m, n);
}

}
