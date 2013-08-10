#pragma once
#include "index.h"

template<typename V, typename F>
bool is_ordered(const V& d,
                F fn) {
    return thrust::equal(d.begin(), d.end(),
                         thrust::make_transform_iterator(
                             thrust::counting_iterator<int>(0),
                             fn));
}

template<typename V>
bool is_tx_row_major(const V& d, int m, int n) {
    return is_ordered(d,
                      inplace::tx_row_major_order
                      <typename V::value_type>(n, m));
}

template<typename V>
bool is_tx_col_major(const V& d, int m, int n) {
    return is_ordered(d, inplace::tx_column_major_order
                      <typename V::value_type>(n, m));
}

template<typename T, typename Fn>
void print_array(const thrust::device_vector<T>& d, Fn index) {
    int m = index.m;
    int n = index.n;
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
