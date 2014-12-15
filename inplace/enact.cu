#include "gcd.h"
#include "introspect.h"
#include "rotate.h"
#include "permute.h"
#include "equations.h"
#include "skinny.h"
#include "util.h"
#include "register_ops.h"
#include <algorithm>


namespace inplace {
namespace detail {


template<typename T, typename F>
__global__ void smem_row_shuffle(int m, int n, T* d, F s);

template<typename T, typename F>
__global__ void memory_row_shuffle(int m, int n, T* d, T* tmp, F s);

template<typename F>
void sm_35_enact(double* data, int m, int n, F s) {
    if (n < 3072) {
        int smem_bytes = sizeof(double) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 4100) {
        register_row_shuffle<double, F, 16>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 16 shuffle");
        
    } else if (n < 6918) {
        register_row_shuffle<double, F, 18>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 18 shuffle");
        
    } else if (n < 30208) {
        register_row_shuffle<double, F, 59>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        double* temp;
        cudaMalloc(&temp, sizeof(double) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename F>
void sm_35_enact(float* data, int m, int n, F s) {
    
    if (n < 6144) {
        int smem_bytes = sizeof(float) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 11326) {
        register_row_shuffle<float, F, 31>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 31 shuffle");
        
    } else if (n < 30720) {
        register_row_shuffle<float, F, 60>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        float* temp;
        cudaMalloc(&temp, sizeof(float) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename F>
void sm_52_enact(double* data, int m, int n, F s) {
    if (n < 6144) {
        int smem_bytes = sizeof(double) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 6918) {
        register_row_shuffle<double, F, 18>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 18 shuffle");
        
    } else if (n < 29696) {
        register_row_shuffle<double, F, 58>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 58 shuffle");
        
    } else {
        double* temp;
        cudaMalloc(&temp, sizeof(double) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename F>
void sm_52_enact(float* data, int m, int n, F s) {
    
    if (n < 12288) {
        int smem_bytes = sizeof(float) * n;
        smem_row_shuffle<<<m, 256, smem_bytes>>>(m, n, data, s);
        check_error("smem shuffle");
    } else if (n < 30720) {
        register_row_shuffle<float, F, 60>
            <<<m, 512>>>(m, n, data, s);
        check_error("register 60 shuffle");
        
    } else {
        float* temp;
        cudaMalloc(&temp, sizeof(float) * n * n_ctas());
        memory_row_shuffle
            <<<n_ctas(), n_threads()>>>(m, n, data, temp, s);
        cudaFree(temp);
        check_error("memory shuffle");
        
    }
}

template<typename T, typename F>
void shuffle_fn(T* data, int m, int n, F s) {
    int arch = current_sm();
    if (arch >= 502) {
        sm_52_enact(data, m, n, s);
    } else if (arch >= 305) {
        sm_35_enact(data, m, n, s);
    } else {
        throw std::invalid_argument("Requires sm_35 or greater");
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
    //Heuristic to choose the fastest implementation
    //based on size of matrix and data layout
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
