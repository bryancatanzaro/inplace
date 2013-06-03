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

template<typename V, typename Fn>
void print_array(const V& d, Fn index) {
    typedef typename V::value_type T;
    int m = index.m;
    int n = index.n;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            T x = d[index(i, j)];
            if (x < 100) {
                std::cout << " ";
            }
            if (x < 10) {
                std::cout << " ";
            }
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
}
