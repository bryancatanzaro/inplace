#pragma once
namespace inplace {
namespace detail {

namespace c2r {

template <typename T>
void skinny_transpose(T* data, int m, int n, T* tmp);

}

namespace r2c {

template <typename T>
void skinny_transpose(T* data, int m, int n, T* tmp);

}

}
}
