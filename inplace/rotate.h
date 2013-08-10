#pragma once
#include <stdio.h>
#include <iostream>
#include <iterator>
#include "index.h"
#include "util.h"
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace inplace {
namespace detail {

template<typename F, typename T>
void rotate(F f, int m, int n, T* data);

}
}
