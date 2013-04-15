#pragma once
#include <stdio.h>

#include "index.h"
namespace inplace {

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

__device__ __forceinline__
unsigned int div_up(unsigned int a, unsigned int b) {
    return (a-1)/b + 1;
}

__device__ __forceinline__
unsigned int div_down(unsigned int a, unsigned int b) {
    return a / b;
}


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
        if (pos >= rm.m_m) pos -= rm.m_m;
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

template<typename T, int U>
__global__ void coarse_col_rotate(int m, int n, T* d) {
    int warp_id = threadIdx.x & 0x1f;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rotation_amount = (global_index - warp_id) % m;
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



template<typename T>
__global__ void fine_col_rotate(int m, int n, T* d) {
    __shared__ T smem[32 * 32]; 

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < n) {
        int warp_id = threadIdx.x & 0x1f;
        int coarse_rotation_amount = (col - warp_id) % m;
        int overall_rotation_amount = col % m;
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

}
