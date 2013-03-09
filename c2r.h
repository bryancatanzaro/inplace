#pragma once

#include "introspect.h"
#include "array.h"
//#include "streaming.h"
#include "index.h"

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
struct gather_col_impl {
    static __device__ void fun(int i, int j, column_major_index cm, const T* d, array<T, R>& s, F& fn) {
        if (i < cm.m_m) {
            s.head = d[cm(fn(i), j)];
        }
        gather_col_impl<T, R-1, F>::fun(i + blockDim.x, j, cm, d, s.tail, fn);
    }
};

template<typename T, typename F>
struct gather_col_impl<T, 1, F> {
    static __device__ void fun(int i, int j, column_major_index cm, const T* d, array<T, 1>& s, F& fn) {
        if (i < cm.m_m) {
            s.head = d[cm(fn(i), j)];
        }
    }
};  

template<typename T, int R, typename F>
__device__ void gather_col(int j, column_major_index cm,
                           const T* d, array<T, R>& s, F& fn) {
    gather_col_impl<T, R, F>::fun(threadIdx.x, j, cm, d, s, fn);
}

template<typename T, int R>
struct write_col_impl {
    static __device__ void fun(const int& i, const int& j,
                               const column_major_index& cm,
                               const array<T, R>& s, T* d) {
        if (i < cm.m_m) {
            //st_glb_cs(d + cm(i, j), s.head);
            d[cm(i, j)] = s.head;
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
            //st_glb_cs(d+cm(i, j), s.head);
            d[cm(i, j)] = s.head;
        }
    }
};  


template<typename T, int R>
__device__ void write_col(const int& j, const column_major_index& cm, 
                      const array<T, R>& s, T* d) {
    write_col_impl<T, R>::fun(threadIdx.x, j, cm, s, d);
}




template<typename T, typename F, int R>
__global__ void inplace_col_op(int m, int n, T* d, F fn) {
    column_major_index cm(m, n);
    //extern __shared__ T storage[];
    array<T, R> thread_storage;

    int j = blockIdx.x;
    fn.set_j(j);
    
        // for(int i = threadIdx.x; i < m; i += blockDim.x) {
        //     //storage[i] = d[cm(i, j)];
        //     storage[i] = ld_glb_cs(d + cm(i, j));
        // }
        // __syncthreads();
        
    gather_col(j, cm, d, thread_storage, fn);
    
    __syncthreads();
    
    write_col(j, cm, thread_storage, d);

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

template<typename T, int R, typename F>
struct gather_row_impl {
    static __device__ void fun(int i, int j, column_major_index cm, const T* d, array<T, R>& s, F& fn) {
        if (j < cm.m_n) {
            s.head = d[cm(i, fn(j))];
        }
        gather_row_impl<T, R-1, F>::fun(i, j + blockDim.x, cm, d, s.tail, fn);
    }
};

template<typename T, typename F>
struct gather_row_impl<T, 1, F> {
    static __device__ void fun(int i, int j, column_major_index cm, const T* d, array<T, 1>& s, F& fn) {
        if (j < cm.m_n) {
            s.head = d[cm(i, fn(j))];
        }
    }
};  

template<typename T, int R, typename F>
__device__ void gather_row(int i, column_major_index cm,
                           const T* d, array<T, R>& s, F& fn) {
    gather_row_impl<T, R, F>::fun(i, threadIdx.x, cm, d, s, fn);
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
__device__ void write_row(const int& i, const column_major_index& cm, 
                          const array<T, R>& s, T* d) {
    write_row_impl<T, R>::fun(i, threadIdx.x, cm, s, d);
}



template<typename T, int R>
__global__ void inplace_row_shuffle(int m, int n, T* d, shuffle s) {
    column_major_index cm(m, n);
    array<T, R> thread_storage;

    int i = blockIdx.x;
    // for(int i = blockIdx.x; i < m; i += gridDim.x) {
    s.set_i(i);
    
    gather_row(i, cm, d, thread_storage, s);
    __syncthreads();
    write_row(i, cm, thread_storage, d);

    // }
}


}
