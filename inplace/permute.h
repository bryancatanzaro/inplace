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

template<typename T>
void postpermute(int m, int n, int c, T* data, int* tmp);

}
}
