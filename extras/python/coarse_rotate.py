from __future__ import print_function

import numpy as np
import random

def print_array(a):
    if len(a.shape) > 1:
        m, n = a.shape
        for row in range(m):
            for col in range(n):
                print('{0: >3}'.format(a[row, col]), end=' ')
            print()

    else:
        n, = a.shape
        for col in range(n):
            print('{0: >3}'.format(a[col]), end=' ')
        print()

def col_array(m, n):
    result = np.ndarray(dtype=np.int32, shape=(m, n))
    for row in range(m):
        for col in range(n):
            result[row, col] = row + col * m
    return result

def coarse_rotate(a, w):
    m, n = a.shape
    result = np.ndarray(dtype=np.int32, shape=(m, n))
    for col in range(0, n, w):
        amount = col % m
        for subcol in range(col, min(col+w, n)):
            for row in range(0, m):
                result[row, subcol] = a[(row + amount) % m, subcol]
    return result

def overall_rotate_amount(m, n):
    return np.array(map(lambda col: col % m, range(n)))

def coarse_rotate_amount(m, n, w):
    return np.array(map(lambda col: ((col//w)* w) % m, range(n)))

def fine_rotate_amount(m, n, w):
    overall = overall_rotate_amount(m, n)
    coarse = coarse_rotate_amount(m, n, w)
    def rotator(col):
        rotation = overall[col] - coarse[col]
        if (rotation < 0):
            return rotation + m
        else:
            return rotation
    return np.array(map(rotator, range(n)))

def test():
    for x in range(1000):
        m = random.randint(1, 10000)
        n = random.randint(1, 10000)
        w = 32
        fine = fine_rotate_amount(m, n, w)
        print("Testing %s x %s" %(m, n), end=" ")
        min_f = min(fine)
        max_f = max(fine)
        print("Min = %s, Max = %s" % (min_f, max_f))
        assert((min_f >= 0) and (max_f < w))
    print("Passed!")

# m = 67
# n = 68
# w = 32
# a = col_array(m, n)
# print_array(a)
# print()
# print()

# b = coarse_rotate(a, w)
# print_array(b)
# print()

# print_array(overall_rotate_amount(a))
# print_array(coarse_rotate_amount(a, w))
# print_array(fine_rotate_amount(a, w))

test()
