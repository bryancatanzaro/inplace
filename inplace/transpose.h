#pragma once

#include "introspect.h"
#include "index.h"
#include "equations.h"

namespace inplace {

namespace c2r {
void transpose(bool row_major, float* data, int m, int n);
void transpose(bool row_major, double* data, int m, int n);
}
namespace r2c {
void transpose(bool row_major, float* data, int m, int n);
void transpose(bool row_major, double* data, int m, int n);
}

void transpose(bool row_major, float* data, int m, int n);
void transpose(bool row_major, double* data, int m, int n);

}

