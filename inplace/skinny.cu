#include "index.h"
#include "gcd.h"
#include "reduced_math.h"
#include "equations.h"
#include "smem.h"
#include <cassert>

namespace inplace {
namespace detail {

namespace c2r {

struct fused_preop {
    reduced_divisor m;
    reduced_divisor b;
    __host__  fused_preop(int _m, int _b) : m(_m), b(_b) {}
    __host__ __device__
    int operator()(const int& i, const int& j) {
        return (int)m.mod(i + (int)b.div(j));
    }
};

struct fused_postop {
    reduced_divisor m;
    int n, c;
    __host__ 
    fused_postop(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    __host__ __device__
    int operator()(const int& i, const int& j) {
        return (int)m.mod(i * n - (int)m.div(i * c) + j);
    }
};


}

namespace r2c {

struct fused_preop {
    reduced_divisor a;
    reduced_divisor c;
    reduced_divisor m;
    int q;
    __host__ 
    fused_preop(int _a, int _c, int _m, int _q) : a(_a) , c(_c), m(_m), q(_q) {}
    __host__ __device__ __forceinline__
    int p(const int& i) {
        int cm1 = (int)c.get() - 1;
        int term_1 = int(a.get()) * (int)c.mod(cm1 * i);
        int term_2 = int(a.mod(int(c.div(cm1+i))*q));
        return term_1 + term_2;
        
    }
    __host__ __device__
    int operator()(const int& i, const int& j) {
        int idx = m.mod(i + (int)m.get() - (int)m.mod(j));
        return p(idx);
    }
};

struct fused_postop {
    reduced_divisor m;
    reduced_divisor b;
    __host__  fused_postop(int _m, int _b) : m(_m), b(_b) {}
    __host__ __device__
    int operator()(const int& i, const int& j) {
        return (int)m.mod(i + (int)m.get() - (int)b.div(j));
    }
};


}


template<typename T, typename F, int U>
__global__ void long_row_shuffle(int m, int n, int i, T* d, T* tmp, F s) {
    row_major_index rm(m, n);
    s.set_i(i);
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
    int j = global_id;
    while(j + U * grid_size < n) {
        #pragma unroll
        for(int k = 0; k < U; k++) {
            tmp[j] = d[rm(i, s(j))];
            j += grid_size;
        }
    }
    while(j < n) {
        tmp[j] = d[rm(i, s(j))];
        j += grid_size;
    }
}

template<typename T, typename F>
__global__ void short_column_permute(int m, int n, T* d, F s) {
    T* smem = shared_memory<T>();
    row_major_index rm(m, n);
    row_major_index blk(blockDim.y, blockDim.x);
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y; // One block tall by REQUIREMENT
    if ((i < m) && (j < n)) {
        smem[blk(i, threadIdx.x)] = d[rm(i, j)];
    }
    __syncthreads();
    if ((i < m) && (j < n)) {
        d[rm(i, j)] = smem[blk(s(i, j), threadIdx.x)];
    }
}

template<typename T, typename F>
void skinny_row_op(F s, int m, int n, T* d, T* tmp) {
    for(int i = 0; i < m; i++) {
        long_row_shuffle<T, F, 4><<<(n-1)/256+1,256>>>(m, n, i, d, tmp, s);
        cudaMemcpy(d + n * i, tmp, sizeof(T) * n, cudaMemcpyDeviceToDevice);
    }
}

template<typename T, typename F>
void skinny_col_op(F s, int m, int n, T* d) {
    int n_threads = 1024 / m;
    n_threads = ((n_threads - 1) / 32 + 1) * 32;
    
    short_column_permute<<<dim3((n-1)/n_threads+1,1), dim3(n_threads, m),
        sizeof(T) * m * n_threads>>>(m, n, d, s);
}


namespace c2r {

template<typename T>
void skinny_transpose(T* data, int m, int n, T* tmp) {

    assert(m <= 32);
    int c, t, k;
    extended_gcd(m, n, c, t);
    if (c > 1) {
        extended_gcd(m/c, n/c, t, k);
    } else {
        k = t;
    }

    if (c > 1) {
        skinny_col_op(fused_preop(m, n/c), m, n, data);
    }
    skinny_row_op(shuffle(m, n, c, k), m, n, data, tmp);
    skinny_col_op(fused_postop(m, n, c), m, n, data);
}


template void skinny_transpose(float* data, int m, int n, float* tmp);
template void skinny_transpose(double* data, int m, int n, double* tmp);
template void skinny_transpose(int* data, int m, int n, int* tmp);
template void skinny_transpose(long long* data, int m, int n, long long* tmp);

}


}
}
