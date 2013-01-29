#pragma once

#include "introspect.h"
#include "array.h"
#include "streaming.h"

namespace inplace {

struct prerotator {
    int m_m, m_n, m_c;
    __host__ __device__
    prerotator(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    int m_a;
    __host__ __device__
    void set_j(const int& j) {
        m_a = j * m_c / m_n;
    }
    __host__ __device__
    int operator()(const int& i) {
        return (i + m_a) % m_m;
    }
};

struct postpermuter {
    int m_m, m_n, m_c, m_j;
    __host__ __device__
    postpermuter(int m, int n, int c) : m_m(m), m_n(n), m_c(c) {}
    __host__ __device__
    void set_j(const int& j) {
        m_j = j;
    }
    __host__ __device__
    int operator()(const int& i) {
        return ((i*m_n)-(i*m_c)/m_m+m_j) % m_m;
    }
};

template<typename T, typename F>
__global__ void col_op(int m, int n, T* d, T* tmp, F fn) {
    column_major_index cm(m, n);
    for(int j = blockIdx.x; j < n; j += gridDim.x) {
        fn.set_j(j); 
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            int src_idx = fn(i);
            tmp[cm(i, blockIdx.x)] = d[cm(src_idx, j)];
        }
        __syncthreads();
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            d[cm(i, j)] = tmp[cm(i, blockIdx.x)];
        }
        __syncthreads();
    }
}

template<typename T, int R, typename F>
struct gather_impl {
    static __device__ void fun(int i, int m, const T* d, array<T, R>& s, F& fn) {
        if (i < m) {
            s.head = d[fn(i)];
        }
        gather_impl<T, R-1, F>::fun(i + blockDim.x, m, d, s.tail, fn);
    }
};

template<typename T, typename F>
struct gather_impl<T, 1, F> {
    static __device__ void fun(int i, int m, const T* d, array<T, 1>& s, F& fn) {
        if (i < m) {
            s.head = d[fn(i)];
        }
    }
};  

template<typename T, int R, typename F>
__device__ void gather(const int& m, const T* d, array<T, R>& s, F& fn) {
    gather_impl<T, R, F>::fun(threadIdx.x, m, d, s, fn);
}

template<typename T, int R>
struct write_col_impl {
    static __device__ void fun(const int& i, const int& j,
                               const column_major_index& cm,
                               const array<T, R>& s, T* d) {
        if (i < cm.m_m) {
            st_glb_cs(d + cm(i, j), s.head);
            //d[cm(i, j)] = s.head;
        }
        write_col_impl<T, R-1>::fun(i + blockDim.x, j, cm, s.tail, d);
    }
};

template<typename T>
struct write_col_impl<T, 1> {
    static __device__ void fun(const int& i, const int& j,
                               const column_major_index& cm,
                               const array<T, 1>& s, T* d) {
        if (i < cm.m_m) {
            st_glb_cs(d+cm(i, j), s.head);
            //d[cm(i, j)] = s.head;
        }
    }
};  


template<typename T, int R>
__device__ void write_col(const column_major_index& cm, const int& j,
                      const array<T, R>& s, T* d) {
    write_col_impl<T, R>::fun(threadIdx.x, j, cm, s, d);
}




template<typename T, typename F, int R>
__global__ void inplace_col_op(int m, int n, T* d, F fn) {
    column_major_index cm(m, n);
    extern __shared__ T storage[];
    array<T, R> thread_storage;

    
    for(int j = blockIdx.x; j < n; j += gridDim.x) {
        fn.set_j(j);
        
        for(int i = threadIdx.x; i < m; i += blockDim.x) {
            //storage[i] = d[cm(i, j)];
            storage[i] = ld_glb_cs(d + cm(i, j));
        }
        __syncthreads();
        
        gather(m, storage, thread_storage, fn);

        write_col(cm, j, thread_storage, d);

        __syncthreads();
    }
}

struct shuffle {
    int m_m, m_n, m_c, m_k;
    __host__ __device__
    shuffle(int m, int n, int c, int k) : m_m(m), m_n(n), m_c(c), m_k(k) {}
    int m_i;
    __host__ __device__
    void set_i(const int& i) {
        m_i = i;
    }
    //This returns long long to avoid integer overflow in intermediate
    //computation
    __host__ __device__
    long long f(const int& j) {
        long long r = j + m_i *(m_n - 1);
        if (m_i < (m_m + 1 - m_c + (j % m_c))) {
            return r;
        } else {
            return r + m_m;
        }
    }
    
    __host__ __device__
    int operator()(const int& j) {
        long long fij = f(j);
        int term1 = (m_k *(fij/m_c)) % (m_n/m_c);
        int term2 = (fij % m_c) * (m_n/m_c);
        return (term1 + term2) % m_n;
    }
};

template<typename T>
__global__ void row_shuffle(int m, int n, T* d, T* tmp, shuffle s) {
    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        column_major_index cm(m, n);
        row_major_index rm(m, n);
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            tmp[rm(blockIdx.x, j)] = d[cm(i, s(j))];
        }
        __syncthreads();
        for(int j = threadIdx.x; j < n; j+= blockDim.x) {
            d[cm(i, j)] = tmp[rm(blockIdx.x, j)];
        }
        __syncthreads();
    }        
}


template<typename T, int R>
struct write_row_impl {
    static __device__ void fun(const int& i, const int& j,
                               const column_major_index& cm,
                               const array<T, R>& s, T* d) {
        if (j < cm.m_n) {
            d[cm(i, j)] = s.head;
        }
        write_row_impl<T, R-1>::fun(i, j + blockDim.x, cm, s.tail, d);
    }
};

template<typename T>
struct write_row_impl<T, 1> {
    static __device__ void fun(const int& i, const int& j,
                               const column_major_index& cm,
                               const array<T, 1>& s, T* d) {
        if (j < cm.m_n) {
            d[cm(i, j)] = s.head;
        }
    }
};  


template<typename T, int R>
__device__ void write_row(const column_major_index& cm, const int& i,
                          const array<T, R>& s, T* d) {
    write_row_impl<T, R>::fun(i, threadIdx.x, cm, s, d);
}



template<typename T, int R>
__global__ void inplace_row_shuffle(int m, int n, T* d, shuffle s) {
    column_major_index cm(m, n);
    row_major_index rm(m, n);
    extern __shared__ T storage[];
    array<T, R> thread_storage;

    for(int i = blockIdx.x; i < m; i += gridDim.x) {
        s.set_i(i);
        for(int j = threadIdx.x; j < n; j += blockDim.x) {
            storage[j] = d[cm(i, j)];
        }
        __syncthreads();
        
        gather(n, storage, thread_storage, s);

        write_row(cm, i, thread_storage, d);

        __syncthreads();
    }
}



template<typename T>
void transpose(bool row_major, int m, int n, T* data, T* tmp_in=0) {
    if (!row_major) {
        std::swap(m, n);
    }

    //temporary_storage<T> tmp(m, n, tmp_in);
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }

    int blockdim = n_ctas();
    int threaddim = n_threads();
    
    if (c > 1) {
        inplace_col_op<T, prerotator, 6><<<blockdim, threaddim, m*sizeof(T)>>>
            (m, n, data, prerotator(m, n, c));
    }
     // row_shuffle<<<blockdim, threaddim>>>(m, n, data, static_cast<T*>(tmp), shuffle(m, n, c, k));
    inplace_row_shuffle<T, 6><<<blockdim, threaddim, n*sizeof(T)>>>
        (m, n, data, shuffle(m, n, c, k));
    inplace_col_op<T, postpermuter, 6><<<blockdim, threaddim, m*sizeof(T)>>>
        (m, n, data, postpermuter(m, n, c));
}

}
