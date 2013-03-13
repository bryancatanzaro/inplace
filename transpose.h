#pragma once

#include "temporary.h"
#include "introspect.h"
#include "index.h"
#include "c2r.h"

namespace inplace {
void transpose(bool row_major, float* data, int m, int n, float* tmp=0);
void transpose(bool row_major, double* data, int m, int n, double* tmp=0);
}

