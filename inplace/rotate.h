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

template<typename T>
void prerotate(int c, int m, int n, T* data);

template<typename T>
void postrotate(int m, int n, T* data);

}
}
