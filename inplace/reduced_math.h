#pragma once

//Dynamically strength-reduced div and mod
//
//Ideas taken from Sean Baxter's MGPU library.
//These classes provide for reduced complexity division and modulus
//on integers, for the case where the same divisor or modulus will
//be used repeatedly.  

namespace inplace {

namespace detail {

// Count leading zeros - start from most significant bit.
__host__ __device__ __forceinline__
int clz(int x) {
#if __CUDA_ARCH__ >= 200
    return __clz(x);
#else
    for(int i = 31; i >= 0; --i)
        if((1<< i) & x) return 31 - i;
    return 32;
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
void find_divisor(unsigned int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff) {
    unsigned int p = 31 + find_log_2(denom, true);
    unsigned int m = ((1ull << p) + denom - 1)/denom;
    mul_coeff = m;
    shift_coeff = p - 32;
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

}


struct reduced_divisor {
    unsigned int mul_coeff;
    unsigned int shift_coeff;
    unsigned int y;
    __host__ __device__ __forceinline__
    reduced_divisor(unsigned int _y) : y(_y) {
        detail::find_divisor(y, mul_coeff, shift_coeff);
    }
    __host__ __device__ __forceinline__
    unsigned int divide(unsigned int x) const {
        return detail::umulhi(x, mul_coeff) >> shift_coeff;
    }
    __host__ __device__ __forceinline__
    unsigned int mod(unsigned int x) const {
        unsigned int quotient = divide(x);
        return x - (quotient * y);
    }
    __host__ __device__ __forceinline__
    unsigned int get() const {
        return y;
    }
};

}

__host__ __device__ __forceinline__
unsigned int operator+(const unsigned int& a,
                       const inplace::reduced_divisor& b) {
    return a + b.get();
}
__host__ __device__ __forceinline__
unsigned int operator+(const inplace::reduced_divisor& a,
                       const unsigned int& b) {
    return a.get() + b;
}
__host__ __device__ __forceinline__
unsigned int operator-(const unsigned int& a,
                       const inplace::reduced_divisor& b) {
    return a - b.get();
}
__host__ __device__ __forceinline__
unsigned int operator-(const inplace::reduced_divisor& a,
                       const unsigned int& b) {
    return a.get() - b;
}

__host__ __device__ __forceinline__
int operator-(const int& a,
              const inplace::reduced_divisor& b) {
    return a - b.get();
}
__host__ __device__ __forceinline__
int operator-(const inplace::reduced_divisor& a,
              const int& b) {
    return a.get() - b;
}

__host__ __device__ __forceinline__
unsigned int operator*(const unsigned int& a,
                       const inplace::reduced_divisor& b) {
    return a * b.get();
}
__host__ __device__ __forceinline__
unsigned int operator*(const inplace::reduced_divisor& a,
                       const unsigned int& b) {
    return a.get() * b;
}
__host__ __device__ __forceinline__
unsigned int operator/(const unsigned int& n,
                       const inplace::reduced_divisor& d) {
    return d.divide(n);
}
__host__ __device__ __forceinline__
unsigned int operator/(const inplace::reduced_divisor& n,
                       const unsigned int& d) {
    return n.get() / d;
}
__host__ __device__ __forceinline__
unsigned int operator%(const unsigned int& n,
                       const inplace::reduced_divisor& d) {
    return d.mod(n);
}
__host__ __device__ __forceinline__
unsigned int operator%(const inplace::reduced_divisor& n,
                       const unsigned int& d) {
    return n.get() % d;
}
