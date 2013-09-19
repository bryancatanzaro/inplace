from __future__ import division, print_function
from fractions import gcd
import numpy as np
import random

def mmi(a, b):
    if b == 1:
        return 0
    if a == 1:
        return 1
    for m in range(b):
        if (a * m) % b == 1:
            return m
    return None

def print_array(a):
    m, n = a.shape
    for row in range(m):
        for col in range(n):
            print('%4s ' % a[row, col], end='')
        print()

def make_row_array(m, n):
    result = np.zeros((m, n), dtype=np.int32)
    for row in range(m):
        for col in range(n):
            result[row, col] = col + row * n
    return result

def make_col_array(m, n):
    result = np.zeros((m, n), dtype=np.int32)
    for row in range(m):
        for col in range(n):
            result[row, col] = row + col * m
    return result

def check_row_array(a):
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if a[i, j] != i * n + j:
                return False
    return True

def check_col_array(a):
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if a[i, j] != j * m + i:
                return False
    return True
                

def row_permute(a, p):
    m, n = a.shape
    result = np.zeros((m, n), dtype=np.int32)
    for col in range(n):
        for row in range(m):
            result[row, col] = a[p[row], col]
    return result

def col_rotate(a, r):
    m, n = a.shape
    result = np.zeros((m, n), dtype=np.int32)
    for col, rotation in zip(range(n), r):
        for row in range(m):
            result[row, col] = a[(row + rotation) % m, col]
    return result

def row_shuffle(a, s):
    m, n = a.shape
    result = np.zeros((m, n), dtype=np.int32)
    for row in range(m):
        for col in range(n):
            result[row, col] = a[row, s[row, col]]
    return result

def properties(a):
    m, n = a.shape
    c = gcd(m, n)
    a = m // c
    b = n // c
    k = mmi(a, b)
    q = mmi(b, a)
    return (m, n, a, b, c, k, q)


def r2c_permutes(a):
    m, n, a, b, c, k, q = properties(a)
    return map(lambda i:
                   (((c-1)*i)% c) * a + \
                   (((c-1+i)//c) * q) % a,
               range(m))

def r2c_prerotates(a):
    m, n = a.shape
    return map(lambda j: m - (j % m), range(n))

def r2c_shuffles(a):
    m, n, a, b, c, k, q = properties(a)
    s = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            s[i, j] = ((i + j // b) % m + j * m) % n
    return s

def r2c_postrotates(a):
    m, n, a, b, c, k, q = properties(a)
    return map(lambda j: m - (j // b), range(n))

def r2c_transpose(a):
    a = row_permute(a, r2c_permutes(a))
    a = col_rotate(a, r2c_prerotates(a))
    a = row_shuffle(a, r2c_shuffles(a))
    a = col_rotate(a, r2c_postrotates(a))
    return a

def c2r_prerotates(a):
    m, n, a, b, c, k, q = properties(a)
    return map(lambda j: j // b, range(n))

def c2r_shuffles(a):
    m, n, a, b, c, k, q = properties(a)
    def f(i, j):
        if i - (j % c) <= m - c:
            return j + i * (n - 1)
        else:
            return j + i * (n - 1) + m
    s = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            fij = f(i, j)
            s[i, j] = (k * (fij // c)) % b + \
                (fij % c) * b
    return s

def c2r_postrotates(a):
    m, n = a.shape
    return map(lambda j: j % m, range(n))

def c2r_permutes(a):
    m, n, a, b, c, k, q = properties(a)
    return map(lambda i: (i * n - i // a) % m, range(m))

def c2r_transpose(a):
    a = col_rotate(a, c2r_prerotates(a))
    a = row_shuffle(a, c2r_shuffles(a))
    a = col_rotate(a, c2r_postrotates(a))
    a = row_permute(a, c2r_permutes(a))
    return a


if __name__ == '__main__' :
    for m in range(1, 100):
        for n in range(1, 100):
            a = make_row_array(m, n)
            print("Testing R2C %s, %s" % (m, n))
            a = r2c_transpose(a)
            #print_array(a)
            assert(check_col_array(a))



    for m in range(1, 100):
        for n in range(1, 100):
            a = make_col_array(m, n)
            print("Testing C2R %s, %s" % (m, n))
            a = c2r_transpose(a)
            #print_array(a)
            assert(check_row_array(a))

    print("Tests passed!")
