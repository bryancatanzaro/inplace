#pragma once

//Dynamically strength-reduced div and mod
//
//Ideas taken from Sean Baxter's MGPU library.
//These classes provide for reduced complexity division and modulus
//on integers, for the case where the same divisor or modulus will
//be used repeatedly.  

namespace inplace {

namespace detail {


void find_divisor(unsigned int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff);


void find_divisor(unsigned long long denom,
                  unsigned long long& mul_coeff, unsigned int& shift_coeff);


__host__ __device__ __forceinline__
unsigned int umulhi(unsigned int x, unsigned int y) {
#if __CUDA_ARCH__ >= 100
    return __umulhi(x, y);
#else
    unsigned long long z = (unsigned long long)x * (unsigned long long)y;
    return (unsigned int)(z >> 32);
#endif  
}


unsigned long long host_umulhi(unsigned long long x, unsigned long long y);

__host__ __device__ __forceinline__
unsigned long long umulhi(unsigned long long x, unsigned long long y) {
#if __CUDA_ARCH__ >= 100
    return __umul64hi(x, y);
#else
    return host_umulhi(x, y);
#endif  
}

}

template<typename U>
struct reduced_divisor_impl {
    U mul_coeff;
    unsigned int shift_coeff;
    U y;
    __host__ __forceinline__
    reduced_divisor_impl(U _y) : y(_y) {
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

typedef reduced_divisor_impl<unsigned int> reduced_divisor;
typedef reduced_divisor_impl<unsigned long long> reduced_divisor_64;


}
