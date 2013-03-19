#pragma once

#include "introspect.h"
#include <new>
#include <iostream>

namespace inplace {
 
template<typename T>
struct temporary_storage {
    T* m_d;
    bool m_owned;
    int m_size;
    temporary_storage(int m, int n, T* tmp=NULL) {
        if (tmp == 0) {
            m_size = max(m, n);
            m_owned = true;
            m_d = NULL;
        } else {
            m_d = tmp;
            m_owned = false;
        }
    }
    ~temporary_storage() {
        if (m_owned && (m_d != NULL))
            cudaFree(m_d);
    }
    operator T*() {
        if (m_d != NULL) {
            return m_d;
        }
        cudaError_t check =
            cudaMalloc(&m_d, n_ctas() * m_size * sizeof(T));
        if (check != cudaSuccess) {
            std::cerr << "Couldn't create temporary storage" << std::endl;
            throw std::bad_alloc();
        }
        return m_d;
    }
};

}
