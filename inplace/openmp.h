#pragma once
#include <omp.h>

namespace inplace {
namespace openmp {

void transpose(bool row_major, float* data, int m, int n, float* tmp);
void transpose(bool row_major, double* data, int m, int n, double* tmp);

}
}
