#pragma once

#include "introspect.h"
#include <new>

namespace inplace {
 
template<typename T>
struct temporary_storage {
    T* m_d;
    bool m_owned;
    temporary_storage(int m, int n, T* tmp) {
        if (tmp == 0) {
            int max_size = max(m, n);
            cudaError_t check =
                cudaMalloc(&m_d, n_ctas() * max_size * sizeof(T));
            if (check != cudaSuccess) {
                std::cerr << "Couldn't create temporary storage" << std::endl;
                throw std::bad_alloc();
            }
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
