#pragma once

#include "temporary.h"
#include "introspect.h"
#include "index.h"
#include "equations.h"

namespace inplace {
namespace c2r {
void transpose(bool row_major, float* data, int m, int n, float* tmp=0);
void transpose(bool row_major, double* data, int m, int n, double* tmp=0);
}
namespace r2c {
void transpose(bool row_major, float* data, int m, int n, float* tmp=0);
void transpose(bool row_major, double* data, int m, int n, double* tmp=0);
}
}

