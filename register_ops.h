#pragma once

namespace inplace {
namespace detail {

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
            d[cm(i, j)] = s.head;
            write_col_impl<T, R-1>::fun(i + blockDim.x, j, cm, s.tail, d);
        }
    }
};

template<typename T>
struct write_col_impl<T, 1> {
    static __device__ void fun(const int& i, const int& j,
                               const column_major_index& cm,
                               const array<T, 1>& s, T* d) {
        if (i < cm.m_m) {
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
__global__ void register_col_op(int m, int n, T* d, F fn) {
    column_major_index cm(m, n);
    array<T, R> thread_storage;

    int j = blockIdx.x;
    fn.set_j(j);
            
    gather_col(j, cm, d, thread_storage, fn);
    
    __syncthreads();
    
    write_col(j, cm, thread_storage, d);

}

template<typename T, int R, typename F>
struct gather_row_impl {
    static __device__ void fun(int i, int j, column_major_index cm, const T* d, array<T, R>& s, F& fn) {
        if (j < cm.m_n) {
            s.head = d[cm(i, fn(j))];
            gather_row_impl<T, R-1, F>::fun(i, j + blockDim.x, cm, d, s.tail, fn);

        }
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
            write_row_impl<T, R-1>::fun(i, j + blockDim.x, cm, s.tail, d);
        }
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
__global__ void register_row_shuffle(int m, int n, T* d, shuffle s) {
    column_major_index cm(m, n);
    array<T, R> thread_storage;

    int i = blockIdx.x;
    s.set_i(i);
    
    gather_row(i, cm, d, thread_storage, s);

    __syncthreads();

    write_row(i, cm, thread_storage, d);

}

}
}
