#pragma once
#include <set>
#include <vector>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <thrust/transform.h>
#include "gcd.h"
#include "index.h"
#include "introspect.h"
#include "util.h"

namespace inplace {
namespace detail {

template<typename T, typename F>
void scatter_permute(F f, int m, int n, T* data, int* tmp);

}
}
