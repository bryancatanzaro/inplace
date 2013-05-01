from __future__ import print_function

import numpy as np
import random

def print_array(a):
    if len(a.shape) > 1:
        m, n = a.shape
        for row in range(m):
            for col in range(n):
                print('{0: >4}'.format(a[row, col]), end=' ')
            print()

    else:
        n, = a.shape
        for col in range(n):
            print('{0: >4}'.format(a[col]), end=' ')
        print()


def triangle(m, n, c):
    result = np.ndarray(shape=(32,32), dtype=np.int32)
    mask = np.ndarray(shape=(32,32), dtype=np.int32)
    for i in range(32):
        for j in range(32):
            col = j + c
            coarse_rotation = ((col//32)*32) % m
            overall_rotation = col % m
            fine_rotation = overall_rotation - coarse_rotation
            if fine_rotation < 0: fine_rotation += m
            triangle_idx = (i + fine_rotation) * 32 + j
            if triangle_idx < (32*32):
                result[i, j] = triangle_idx
                mask[i, j] = 1
            else:
                result[i, j] = triangle_idx - 32*32
                mask[i, j] = -1
    return result, mask

def populate(tri):
    result = np.ndarray(shape=(32*32), dtype=np.int32)
    for i in range(32):
        for j in range(32):
            if tri[i, j] > -1:
                result[tri[i, j]] = tri[i, j]
    return result.reshape(32,32)

a, mask = triangle(52, 100, 35)
print_array(a)
print_array(mask)
