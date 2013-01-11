#pragma once

namespace inplace {

int n_ctas() {
    //XXX This should scale with GPU size
    return 2 * 14;
}

template<typename T>
struct temporary_storage {
    T* m_d;
    bool m_owned;
    temporary_storage(int m, int n, T* tmp) {
        if (tmp == 0) {
            int max_size = max(m, n);
            cudaMalloc(&m_d, n_ctas() * max_size * sizeof(T));
            m_owned = true;
        } else {
            m_d = tmp;
            m_owned = false;
        }
    }
    ~temporary_storage() {
        if (m_owned)
            cudaFree(m_d);
    }
    operator T*() {
        return m_d;
    }
};

}
