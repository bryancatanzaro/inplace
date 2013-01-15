#pragma once
namespace inplace {

template<typename T>
struct column_major_order {
    typedef T result_type;

    int m_m;
    int m_n;

    __host__ __device__
    column_major_order(const int& m, const int& n) :
        m_m(m), m_n(n) {}
    
    __host__ __device__ T operator()(const int& idx) {
        int row = idx % m_m;
        int col = idx / m_m;
        return row * m_m + col;
    }
};

template<typename T>
struct row_major_order {
    typedef T result_type;

    int m_m;
    int m_n;

    __host__ __device__
    row_major_order(const int& m, const int& n) :
        m_m(m), m_n(n) {}

    __host__ __device__ T operator()(const int& idx) {
        int row = idx % m_n;
        int col = idx / m_n;
        return col * m_n + row;
    }
};

template<typename T>
struct tx_column_major_order {
    typedef T result_type;

    int m_m;
    int m_n;

    __host__ __device__
    tx_column_major_order(const int& m, const int& n) :
        m_m(m), m_n(n) {}
    
    __host__ __device__ T operator()(const int& idx) {
        int row = idx % m_m;
        int col = idx / m_m;
        return col * m_n + row;
    }
};

template<typename T>
struct tx_row_major_order {
    typedef T result_type;

    int m_m;
    int m_n;

    __host__ __device__
    tx_row_major_order(const int& m, const int& n) :
        m_m(m), m_n(n) {}

    __host__ __device__ T operator()(const int& idx) {
        int row = idx % m_n;
        int col = idx / m_n;
        return row * m_m + col;
    }
};

struct column_major_index {
    int m_m;
    int m_n;

    __host__ __device__
    column_major_index(const int& m, const int& n) :
        m_m(m), m_n(n) {}
    
    __host__ __device__ int operator()(const int& i, const int& j) {
        return i + j * m_m;
    }
};

struct row_major_index {
    int m_m;
    int m_n;

    __host__ __device__
    row_major_index(const int& m, const int& n) :
        m_m(m), m_n(n) {}

    __host__ __device__ int operator()(const int& i, const int& j) {
        return j + i * m_n;
    }
};


}
