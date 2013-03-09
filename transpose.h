#pragma once

#include "temporary.h"
#include "introspect.h"
#include "index.h"
#include "c2r.h"

namespace inplace {
void transpose(bool row_major, float* data, int m, int n);
void transpose(bool row_major, double* data, int m, int n);
}

// template<typename T>
// void transpose(bool row_major, int m, int n, T* data, T* tmp_in=0) {
//     if (!row_major) {
//         int o = m;
//         m = n;
//         n = o;
//     }   

    
//     //temporary_storage<T> tmp(m, n, tmp_in);
//     int c, t, k;
//     extended_gcd(m, n, c, t);
//     if (c > 1) {
//         extended_gcd(m/c, n/c, t, k);
//     } else {
//         k = t;
//     }

//     int blockdim_col = n;
//     int blockdim_row = m;
//     // int blockdim = n_ctas();
//     int threaddim = 512;//n_threads();

//     #define WPT 190
//     //Verified to work on SM_35
//     //with no spills/fills for row_shuffle
    
//     if (c > 1) {
//         inplace_col_op<T, prerotator, WPT><<<blockdim_col, threaddim>>>
//             (m, n, data, prerotator(m, n, c));
//     }
    
//     inplace_row_shuffle<T, WPT><<<blockdim_row, threaddim>>>
//         (m, n, data, shuffle(m, n, c, k));

//     inplace_col_op<T, postpermuter, WPT><<<blockdim_col, threaddim>>>
//         (m, n, data, postpermuter(m, n, c));

// }
