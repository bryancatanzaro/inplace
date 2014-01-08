namespace inplace {
namespace detail {

void find_divisor(unsigned long long denom,
                  unsigned long long& mul_coeff, unsigned int& shift_coeff) {
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

void cpu_umulhi(unsigned long long x, unsigned long long y) {
    __uint128_t z = __uint128_t(x) * __uint128_t(y);
    return (unsigned long long)(z >> 64);
}


}
}
