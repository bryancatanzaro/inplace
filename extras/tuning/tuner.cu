#include "schedule.h"
#include "sm.h"
#include "memory_ops.h"
#include "register_ops.h"
#include "smem_ops.h"
#include "timer.h"
#include "gcd.h"
#include "equations.h"
#include <iostream>

#ifdef SMEM
#define VARIANT smem<float, SM, BLKS>
#endif

#ifdef REG
#define VARIANT reg<WPT, -1>
#endif

namespace inplace {
namespace detail {

template<typename T>
struct launcher{};

template<typename T, typename SM, int blks>
struct launcher<smem<T, SM, blks> > {
    static void impl(int min, int max) {
        timer the_timer;
        int m = 200;
        int tpb = smem<T, SM, blks>::blk;
        T* data;
        cudaMalloc(&data, sizeof(T) * m * max);
        for(int n = min; n < max; n++) {
            int c, t, k;
            extended_gcd(m, n, c, t);
            if (c > 1) {
                extended_gcd(m/c, n/c, t, k);
            } else {
                k = t;
            }
            r2c::shuffle f(m, n, c, k);
            float time;
            if (n < smem<T, SM, blks>::lim) {
                int smem_bytes = sizeof(T) * n;
                the_timer.start();
                smem_row_shuffle<<<m, tpb, smem_bytes>>>(m, n, data, f);
                time = the_timer.stop();
                float bw = n * m * 2 * sizeof(T) / time / 1e6;
                std::cout << bw << " ";
            } else {
                std::cout << "0 ";
            }
            
        }
        cudaFree(data);
    }
};

template<int wpt>
struct launcher<reg<wpt, -1> > {
    static void impl(int min, int max) {
        timer the_timer;
        int m = 200;
        typedef float T;
        T* data;
        cudaMalloc(&data, sizeof(T) * m * max);
        int tpbs[] = {512};
        for(int tpb_i = 0; tpb_i < 1; tpb_i++) {
            for(int n = min; n < max; n++) {
                int c, t, k;
                extended_gcd(m, n, c, t);
                if (c > 1) {
                    extended_gcd(m/c, n/c, t, k);
                } else {
                    k = t;
                }
                r2c::shuffle f(m, n, c, k);
                float time;
                if (n < wpt * tpbs[tpb_i]) {
                    the_timer.start();
                    register_row_shuffle<SM, float, r2c::shuffle, wpt> 
                        <<<m, tpbs[tpb_i]>>>(m, n, data, f);
                    time = the_timer.stop();
                    cudaError_t err = cudaGetLastError();
                    if (err == cudaSuccess) {
                        float bw = n * m * 2 * sizeof(T) / time / 1e6;
                        std::cout << bw << " ";
                    } else {
                        std::cout << "0 ";
                    }
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << std::endl;
        }
        cudaFree(data);
    }
};



void benchmark(int min, int max) {
    launcher<VARIANT>::impl(min, max);

}

}
}

int main() {
    inplace::detail::benchmark(1, 29440);
}
    
