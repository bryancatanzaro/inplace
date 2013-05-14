#pragma once

namespace inplace {
namespace sequential {

void transpose(bool row_major, float* data, int m, int n, float* tmp);
void transpose(bool row_major, double* data, int m, int n, double* tmp);

}
}
