#include <stdexcept>

namespace inplace {
namespace detail {

// Count leading zeros - start from most significant bit.
int clz(int x) {
    for(int i = 31; i >= 0; --i)
        if((1<< i) & x) return 31 - i;
    return 32;
}

// Count leading zeros - start from most significant bit.
int clz(long long x) {
    for(int i = 63; i >= 0; --i)
        if((1ll<< i) & x) return 63 - i;
    return 32;
}


#define INPLACE_IS_POW_2(x) (0 == ((x) & ((x) - 1)))

int find_log_2(int x, bool round_up = false) {
    int a = 31 - clz(x);
    if (round_up) a += !INPLACE_IS_POW_2(x);
    return a;
}

int find_log_2(long long x, bool round_up = false) {
    int a = 63 - clz(x);
    if (round_up) a += !INPLACE_IS_POW_2(x);
    return a;
}


void find_divisor(unsigned int denom,
                  unsigned int& mul_coeff, unsigned int& shift_coeff) {
    if (denom == 0) {
        throw std::invalid_argument("Trying to find reduced divisor for 0");
    }
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


void find_divisor(unsigned long long denom,
                  unsigned long long& mul_coeff, unsigned int& shift_coeff) {
    if (denom == 0) {
        throw std::invalid_argument("Trying to find reduced divisor for 0");
    }
    if (denom == 1) {
        mul_coeff = 0;
        shift_coeff = 0;
        return;
    }
    unsigned int p = 63 + find_log_2((long long)denom, true);
    unsigned long long m = (((__uint128_t(1) << p) + denom - 1)/denom);
    mul_coeff = m;
    shift_coeff = p - 64;
}

unsigned long long host_umulhi(unsigned long long x, unsigned long long y) {
    __uint128_t z = __uint128_t(x) * __uint128_t(y);
    return (unsigned long long)(z >> 64);
}


}
}
