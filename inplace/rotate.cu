#include "rotate.h"


namespace inplace {
namespace detail {

__device__ __forceinline__
unsigned int ctz(unsigned int x) {
    return __ffs(x) - 1;
}

__device__ __forceinline__
unsigned int gcd(unsigned int x, unsigned int y) {
    if (x == 0) return y;
    if (y == 0) return x;
    unsigned int cf2 = ctz(x | y);
    x >>= ctz(x);
    while (true) {
        y >>= ctz(y);
        if (x == y) break;
        if (x > y) {
            unsigned int t = x; x = y; y = t;
        }
        if (x == 1) break;
        y -= x;
    }
    return x << cf2;
}


struct prerotate_fn {
    typedef int result_type;
    int b;
    __host__ __device__ prerotate_fn(int _b) : b(_b) {}
    __host__ __device__
    int operator()(int j) const {
        return div_down(j, b);
    }
    __host__ __device__
    bool fine() const {
        return (b % 32) != 0;
    }
};


struct postrotate_fn {
    int m;
    __host__ __device__ postrotate_fn(int _m) : m(_m) {}
    __host__ __device__
    int operator()(int j) const {
        return j % m;
    }
    __host__ __device__
    bool fine() const {
        return true;
    }
};


template<typename T, int U>
__device__ __forceinline__
void unroll_rotate(T& prior, int& pos, int col, row_major_index rm, int inc, T* d) {
    T tmp[U];
    int positions[U];
    //Compute positions
#pragma unroll
    for(int i = 0; i < U; i++) {
        positions[i] = pos;
        pos += inc;
        if (pos >= rm.m) pos -= rm.m;
    }
    //Load temporaries
#pragma unroll
    for(int i = 0; i < U; i++) {
        tmp[i] = d[rm(positions[i], col)];
    }
    //Store results
    d[rm(positions[0], col)] = prior;
#pragma unroll
    for(int i = 0; i < U-1; i++) {
        d[rm(positions[i+1], col)] = tmp[i];
    }
    prior = tmp[U-1];

}

template<typename F, typename T, int U>
__global__ void coarse_col_rotate(F fn, int m, int n, T* d) {
    int warp_id = threadIdx.x & 0x1f;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = fn(global_index - warp_id);
    int col = global_index;
    
    if ((col < n) && (rotation_amount > 0)) {
        row_major_index rm(m, n);
        int c = gcd(rotation_amount, m);
        int l = m / c;
        int inc = m - rotation_amount;
        for(int b = 0; b < c; b++) {
            int pos = b;
            T prior = d[rm(pos, col)];
            pos += inc;
            if (pos >= m)
                pos -= m;
            int x = 0;
            for(; x < l - U + 1; x += U) {
                unroll_rotate<T, U>(prior, pos, col, rm, inc, d);
            }
            for(; x < l; x++) {
                unroll_rotate<T, 1>(prior, pos, col, rm, inc, d);
            }
            d[rm(pos, col)] = prior;

        }
    }
}



template<typename F, typename T>
__global__ void fine_col_rotate(F fn, int m, int n, T* d) {
    __shared__ T smem[32 * 32]; 

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < n) {
        int warp_id = threadIdx.x & 0x1f;
        int coarse_rotation_amount = fn(col - warp_id);
        int overall_rotation_amount = fn(col);
        int fine_rotation_amount = overall_rotation_amount - coarse_rotation_amount;
        if (fine_rotation_amount < 0) fine_rotation_amount += m;

        int row = threadIdx.y;
        int idx = row * n + col;
        T* read_ptr = d + idx;
        
        int smem_idx = threadIdx.y * 32 + threadIdx.x;

        T first = -2;
        if (row < m) first = *read_ptr;

        bool first_phase = (threadIdx.y >= fine_rotation_amount);
        int smem_row = threadIdx.y - fine_rotation_amount;
        if (!first_phase) smem_row += 32;

        int smem_write_idx = smem_row * 32 + threadIdx.x;

        if (first_phase) smem[smem_write_idx] = first;

        T* write_ptr = read_ptr;
        int ptr_inc = 32 * n;
        read_ptr += ptr_inc;
        //Loop over blocks that are guaranteed not to fall off the edge
        for(int i = 0; i < (m / 32) - 1; i++) {
            T tmp = *read_ptr;
            if (!first_phase) smem[smem_write_idx] = tmp;
            __syncthreads();
            *write_ptr = smem[smem_idx];
            __syncthreads();
            if (first_phase) smem[smem_write_idx] = tmp;
            write_ptr = read_ptr;
            read_ptr += ptr_inc;
        }

        //Final block (read_ptr may have fallen off the edge)
        int remainder = m % 32;
        T tmp = -3;
        if (threadIdx.y < remainder) tmp = *read_ptr;
        int tmp_dest_row = 32 - fine_rotation_amount + threadIdx.y;
        if ((tmp_dest_row >= 0) && (tmp_dest_row < 32))
            smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
        __syncthreads();
        int first_dest_row = 32 + remainder - fine_rotation_amount + threadIdx.y;
        if ((first_dest_row >= 0) && (first_dest_row < 32))
            smem[first_dest_row * 32 + threadIdx.x] = first;
        
        __syncthreads();
        *write_ptr = smem[smem_idx];
        write_ptr = read_ptr;
        __syncthreads();
        //Final incomplete block
        tmp_dest_row -= 32; first_dest_row -= 32;
        if ((tmp_dest_row >= 0) && (tmp_dest_row < 32))
            smem[tmp_dest_row * 32 + threadIdx.x] = tmp;
        __syncthreads();
        if ((first_dest_row >= 0) && (first_dest_row < 32))
            smem[first_dest_row * 32 + threadIdx.x] = first;
        __syncthreads();
        if (threadIdx.y < remainder) *write_ptr = smem[smem_idx];
    }
}

template<typename F, typename T>
void full_rotate(F fn, int m, int n, T* data) {
    if (fn.fine()) {
        fine_col_rotate<<<div_up(n, 32), dim3(32,32)>>>(fn, m, n, data);
    }
    int block_size = 256;
    int n_blocks = div_up(n, block_size);
    coarse_col_rotate<F, T, 4><<<n_blocks, block_size>>>(
        fn, m, n, data);
}




template<typename T>
void prerotate(int c, int m, int n, T* data) {
    full_rotate(prerotate_fn(n/c), m, n, data);
}


template<typename T>
void postrotate(int m, int n, T* data) {
    full_rotate(postrotate_fn(m), m, n, data);
}


template void prerotate<float>(int, int, int, float*);
template void prerotate<double>(int, int, int, double*);
template void prerotate<int>(int, int, int, int*);
template void prerotate<long long>(int, int, int, long long*);


template void postrotate<float>(int, int, float*);
template void postrotate<double>(int, int, double*);
template void postrotate<int>(int, int, int*);
template void postrotate<long long>(int, int, long long*);



}
}
