#include "permute.h"
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <cassert>

struct gather_permute {
    int m, n, c;
    __host__ __device__
    gather_permute(int _m, int _n, int _c) : m(_m), n(_n), c(_c) {}
    __host__ __device__
    int operator()(const int& i) {
        return (i * n - (i * c)/m) % m;
    }
};

struct golden_permute {
    typedef int result_type;
    gather_permute gather;
    __host__ __device__
    golden_permute(int _m, int _n, int _c) : gather(_m, _n, _c) {}
    __host__ __device__
    int operator()(const int& idx) {
        int i = idx / gather.n;
        int j = idx % gather.n;
        int source_i = gather(i);
        return source_i * gather.n + j;
    }
};

template<typename T, typename Fn>
void print_array(const thrust::device_vector<T>& d, Fn index) {
    int m = index.m_m;
    int n = index.m_n;
    thrust::host_vector<T> h = d;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = h[index(i, j)];
            std::cout.width(5); std::cout << std::right;
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    //int m = 6; int n = 32;
    int m = 999; int n = 100000;
    int c, q;
    inplace::extended_gcd(m, n, c, q);
    thrust::device_vector<int> data(m * n);
    thrust::counting_iterator<int> x(0);
    thrust::copy(x, x+(m*n), data.begin());
    thrust::device_vector<int> tmp(m);

    inplace::detail::postpermute(m, n, c,
                                 thrust::raw_pointer_cast(data.data()),
                                 thrust::raw_pointer_cast(tmp.data()));
    assert(thrust::equal(
               data.begin(), data.end(),
               thrust::make_transform_iterator(x, golden_permute(m, n, c))));

    //print_array(data, inplace::row_major_index(m, n));
}
