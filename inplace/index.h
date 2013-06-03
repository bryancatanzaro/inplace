#pragma once
namespace inplace {

template<typename T>
struct column_major_order {
    typedef T result_type;

    int m;
    int n;

    __host__ __device__
    column_major_order(const int& _m, const int& _n) :
        m(_m), n(_n) {}
    
    __host__ __device__ T operator()(const int& idx) const {
        int row = idx % m;
        int col = idx / m;
        return row * m + col;
    }
};

template<typename T>
struct row_major_order {
    typedef T result_type;

    int m;
    int n;

    __host__ __device__
    row_major_order(const int& _m, const int& _n) :
        m(_m), n(_n) {}

    __host__ __device__ T operator()(const int& idx) const {
        int row = idx % n;
        int col = idx / n;
        return col * n + row;
    }
};

template<typename T>
struct tx_column_major_order {
    typedef T result_type;

    int m;
    int n;

    __host__ __device__
    tx_column_major_order(const int& _m, const int& _n) :
        m(_m), n(_n) {}
    
    __host__ __device__ T operator()(const int& idx) const {
        int row = idx / m;
        int col = idx % m;
        return col * n + row;
    }
};

template<typename T>
struct tx_row_major_order {
    typedef T result_type;

    int m;
    int n;

    __host__ __device__
    tx_row_major_order(const int& _m, const int& _n) :
        m(_m), n(_n) {}

    __host__ __device__ T operator()(const int& idx) const {
        int row = idx % n;
        int col = idx / n;
        return row * m + col;
    }
};

struct column_major_index {
    int m;
    int n;

    __host__ __device__
    column_major_index(const int& _m, const int& _n) :
        m(_m), n(_n) {}
    
    __host__ __device__ int operator()(const int& i, const int& j) const {
        return i + j * m;
    }
};

struct row_major_index {
    int m;
    int n;

    __host__ __device__
    row_major_index(const int& _m, const int& _n) :
        m(_m), n(_n) {}

    __host__ __device__ int operator()(const int& i, const int& j) const {
        return j + i * n;
    }
};


}
