#pragma once

//Dynamically strength-reduced div and mod
//
//Ideas taken from Sean Baxter's MGPU library.
//These classes provide for reduced complexity division and modulus
//on integers, for the case where the same divisor or modulus will
//be used repeatedly.  
#include <iostream>

namespace inplace {

namespace detail {

// Count leading zeros - start from most significant bit.
__host__ __device__ __forceinline__
int clz(int x) {
#if __CUDA_ARCH__ >= 100
    return __clz(x);
#else
    for(int i = 31; i >= 0; --i)
        if((1<< i) & x) return 31 - i;
    return 32;
#endif
}

// Count leading zeros - start from most significant bit.
__host__ __device__ __forceinline__
int clz(long long x) {
#if __CUDA_ARCH__ >= 100
    return __clzll(x);
#else
    for(int i = 63; i >= 0; --i)
        if((1ll<< i) & x) return 63 - i;
    return 64;
#endif
}

#define INPLACE_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

__host__ __device__ __forceinline__
int find_log_2(int x, bool round_up = false) {
    int a = 31 - clz(x);
    if (round_up) a += !INPLACE_IS_POW_2(x);
    return a;
}

__host__ __device__ __forceinline__
int find_log_2(long long x, bool round_up = false) {
    int a = 63 - clz(x);
    if (round_up) a += !INPLACE_IS_POW_2(x);
    return a;
}

__host__ __device__ __forceinline__
void find_divisor(unsigned int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff) {
    if (denom == 1) {
        mul_coeff = 0;
        shift_coeff = 0;
        return;
    }
    unsigned int p = 31 + find_log_2((int)denom, true);
    unsigned int m = ((1ull << p) + denom - 1)/denom;
    mul_coeff = m;
    shift_coeff = p - 32;
}

__host__ __forceinline__
void find_divisor(unsigned long long denom,
                  unsigned long long& mul_coeff, unsigned int& shift_coeff) {
    if (denom == 1) {
        mul_coeff = 0;
        shift_coeff = 0;
    }
    unsigned int p = 63 + find_log_2((long long)denom, true);
    typedef unsigned int uint128_t __attribute__((mode(TI)));
    unsigned long long m = (((uint128_t)1 << p) + denom - 1)/denom;
    mul_coeff = m;
    shift_coeff = p - 64;
}

__host__ __device__ __forceinline__
unsigned int umulhi(unsigned int x, unsigned int y) {
#if __CUDA_ARCH__ >= 100
    return __umulhi(x, y);
#else
    unsigned long long z = (unsigned long long)x * (unsigned long long)y;
    return (unsigned int)(z >> 32);
#endif  
}

__host__ __device__ __forceinline__
unsigned long long umulhi(unsigned long long x, unsigned long long y) {
#if __CUDA_ARCH__ >= 100
    return __umul64hi(x, y);
#else
    typedef unsigned int uint128_t __attribute__((mode(TI)));
    return ((uint128_t)x * y) >> 64;
#endif  
}

}

template<typename U>
struct reduced_divisor {
    U mul_coeff;
    unsigned int shift_coeff;
    U y;
    __host__ __forceinline__
    reduced_divisor(U _y) : y(_y) {
        detail::find_divisor(y, mul_coeff, shift_coeff);
    }
    __host__ __device__ __forceinline__
    U div(U x) const {
        return (mul_coeff) ? detail::umulhi(x, mul_coeff) >> shift_coeff : x;
    }
    __host__ __device__ __forceinline__
    U mod(U x) const {
        return (mul_coeff) ? x - (div(x) * y) : 0;
    }
    __host__ __device__ __forceinline__
    void divmod(U x, U& q, U& mod) {
        if (y == 1) {
            q = x; mod = 0;
        } else {
            q = div(x);
            mod = x - (q * y);
        }
    }   
    __host__ __device__ __forceinline__
    U get() const {
        return y;
    }
};

typedef reduced_divisor<unsigned int> reduced_divisor_32;
typedef reduced_divisor<unsigned long long> reduced_divisor_64;

}
