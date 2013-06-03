#include <set>
#include <vector>
#include <thrust/iterator/counting_iterator.h>
#include "gcd.h"
#include "index.h"
#include <iostream>
#include <thrust/transform.h>
#include "introspect.h"

namespace inplace {
namespace detail {

struct scatter_permutes {
    typedef int result_type;
    int m; int n; int c; int a; int b; int q;
    __host__
    scatter_permutes(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {
        int d;
        extended_gcd(n/c, m/c, d, q);
        a = m / c;
        b = n / c;
    }
    __host__ __device__
    int operator()(int i) const {
        int k = ((c - 1) * i) % c;
        int l = ((c - 1 + i) / c);
        int r = k * a + ((l * q) % a);
        return r;
    }
    __host__ __device__
    int len() const {
        return m;
    }
};
    
template<typename Fn>
std::vector<int> scatter_cycles(Fn f) {
    int len = f.len();
    thrust::counting_iterator<int> i(0);
    std::set<int> unvisited(i, i+len);
    std::vector<int> heads;
    while(!unvisited.empty()) {
        int idx = *unvisited.begin();
        unvisited.erase(unvisited.begin());
        int dest = f(idx);
        if (idx != dest) {
            heads.push_back(idx);
            int start = idx;
            while(dest != start) {
                idx = dest;
                unvisited.erase(idx);
                dest = f(idx);
            }
        }
    }
    return heads;
}


template<typename T, int U>
__device__ __forceinline__ void unroll_cycle_row_permute(
    scatter_permutes f, row_major_index rm, T* data, int start, int j) {
    T src[U];
    int inc = gridDim.x * blockDim.x;
    int index = start;
    T* src_ptr = data + rm(index, j);
    #pragma unroll
    for(int i = 0; i < U; i++) {
        src[i] = *src_ptr;
        src_ptr += inc;
    }
    do {
        index = f(index);
        src_ptr = data + rm(index, j);
        T dest[U];
        T* load_ptr = src_ptr;
        #pragma unroll
        for(int i = 0; i < U; i++) {
            dest[i] = *load_ptr;
            load_ptr += inc;
        }
        #pragma unroll
        for(int i = 0; i < U; i++) {
            *src_ptr = src[i];
            src[i] = dest[i];
            src_ptr += inc;
        }
    } while (index != start);
}

template<typename T, int U>
__global__ void cycle_row_permute(scatter_permutes f, T* data, int* heads,
    int n_heads) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int n = f.n;
    row_major_index rm(f.m, f.n);

    int warp_id = threadIdx.x & 0x1f;
    int eow = 31 - warp_id;
    int inc = gridDim.x * blockDim.x;


    while((j + (U-1) * inc + eow) < n) {
        for(int h = 0; h < n_heads; h++) {
            int i = heads[h];
            unroll_cycle_row_permute<T, U>(f, rm, data, i, j);
        }
        j += inc * U;
    }
    while(j < n) {
        for(int h = 0; h < n_heads; h++) {
            int i = heads[h];
            unroll_cycle_row_permute<T, 1>(f, rm, data, i, j);
        }
        j += inc;
    }
}


template<typename T>
void postpermute(int m, int n, int c, T* data, int* tmp) {
    scatter_permutes f(m, n, c);
    std::vector<int> heads = scatter_cycles(f);
    cudaMemcpy(tmp, heads.data(), sizeof(int)*heads.size(),
               cudaMemcpyHostToDevice);
    int n_blocks = 14*8;//n_ctas();
    int n_threads = 256;
    cycle_row_permute<T, 4><<<n_blocks, n_threads>>>
        (f, data, tmp, heads.size());
    
}

}
}

