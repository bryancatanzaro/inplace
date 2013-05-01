import numpy as np
from fractions import gcd
import itertools

class Array(object):
    def __init__(self, *shape):
        self.d = np.ndarray(shape=shape, dtype=np.int32)
    def __str__(self):
        m, n = self.d.shape
        result = ""
        for row in range(m):
            for col in range(n):
                el = self.d[row, col]
                if el < 10:
                    result += ' '
                if el < 100:
                    result += ' '
                result += str(el) + ' '
            result += '\n'
        return result
    def __getitem__(self, key):
        return self.d[key]
    def __setitem__(self, key, val):
        self.d[key] = val
    @property
    def shape(self):
        return self.d.shape
    @shape.setter
    def shape(self, val):
        self.d.shape = val
        
def is_power_of_two(x):
    return (x & (x - 1)) == 0

def is_odd(x):
    return (x % 2 == 1)

    
def mmi(a, b):
    for m in range(b):
        if (a * m) % b == 1:
            return m
    return None

def make_row_array(m, n):
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            result[row, col] = col + row * n
    return result

def make_col_array(m, n):
    result = Array(m, n)
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
    result = Array(m, n)
    for col in range(n):
        for row in range(m):
            result[row, col] = a[p[row], col]
    return result

def col_rotate(a, r):
    m, n = a.shape
    result = Array(m, n)
    for col, rotation in zip(range(n), r):
        for row in range(m):
            result[row, col] = a[(row + rotation) % m, col]
    return result

def row_shuffle(a, s):
    m, n = a.shape
    result = Array(m, n)
    for row in range(m):
        for col in range(n):
            result[row, col] = a[row, s[row, col]]
    return result

def c2r_odd_shuffles(m, n):
    result = Array(m, n)
    for i in range(m):
        for j in range(n):
            result[i, j] = ((i - j + n)*(n - mmi(m, n))) % n
    return result

def c2r_odd_permutes(m, n):
    return map(lambda i: (i * mmi(mmi(n, m), m)) % m, range(m))

def c2r_odd_rotates(m, n):
    return map(lambda j: (j * mmi(n, m)) % m, range(n))

def c2r_odd_transpose(a):
    m, n = a.shape
    if not is_power_of_two(n) or \
            not is_odd(m) or \
            not m > 1 or \
            not n > 1:
        raise ValueError("Constraints not satisfied")
    print("Testing C2R Odd Transpose. m, n = (%s, %s)" %(m, n))
    shuffled = row_shuffle(a, c2r_odd_shuffles(m, n))
    permuted = row_permute(shuffled, c2r_odd_permutes(m, n))
    rotated = col_rotate(permuted, c2r_odd_rotates(m, n))
    return rotated

def c2r_po2_prerotates(m, n):
    return map(lambda j: (j * m)//n, range(n))

def c2r_po2_shuffles(m, n):
    result = Array(m, n)
    for i in range(m):
        for j in range(n):
            result[i, j] = ((j*(n+1))//m + (i*(m-1)*n)/m) % n
    return result

def c2r_po2_permutes(m, n):
    return map(lambda i: (i * (m-1)) % m, range(m))

def c2r_po2_rotates(m, n):
    return map(lambda j: (j * (m-1)) % m, range(n))

def c2r_po2_transpose(a):
    m, n = a.shape
    c = gcd(m, n)
    if not is_power_of_two(n) or not c == m:
        raise ValueError("Constraints not satisfied")
    print("Testing C2R Po2 Transpose. m, n = (%s, %s)" %(m, n))
    prerotated = col_rotate(a, c2r_po2_prerotates(m, n))
    shuffled = row_shuffle(prerotated, c2r_po2_shuffles(m, n))
    permuted = row_permute(shuffled, c2r_po2_permutes(m, n))
    rotated = col_rotate(permuted, c2r_po2_rotates(m, n))
    return rotated

def c2r_comp_prerotates(m, n):
    c = gcd(m, n)
    return map(lambda j: (j * c)//n, range(n))

def c2r_comp_shuffles(m, n):
    def f(i, j):
        c = gcd(m, n)
        if (i < (m + 1 - c + (j % c))):
            return j + i * (n-1)
        else:
            return j + i * (n-1) + m

    result = Array(m, n)
    c = gcd(m, n)
    k = mmi(m/c, n/c)
    if not k:
        k = 0
    for i in range(m):
        for j in range(n):
            fij = f(i, j)
            result[i, j] = (((fij//c)*k) % (n/c) + (fij % c)*(n/c)) % n
    return result

def c2r_comp_rotates(m, n):
    return map(lambda j: j % m, range(n))

def c2r_comp_permutes(m, n):
    c = gcd(m, n)
    a = map(lambda i: (i * n - ((i * c )// m)) % m, range(m))
    return a

def c2r_comp_transpose(a):
    m, n = a.shape
    c = gcd(m, n)
    #if not is_power_of_two(n) or not c < n or not c < m:
    #    raise ValueError("Constraints not satisfied")
    
    print("Testing C2R Cmp Transpose. m, n = (%s, %s)" %(m, n))
    #np.save("golden_begin", a.d)
    prerotated = col_rotate(a, c2r_comp_prerotates(m, n))
    #np.save("golden_prerotate", prerotated.d)
    shuffled = row_shuffle(prerotated, c2r_comp_shuffles(m, n))
    np.save("golden_shuffle_idxes", c2r_comp_shuffles(m, n).d)
    #np.save("golden_shuffle", shuffled.d)
    rotated = col_rotate(shuffled, c2r_comp_rotates(m, n))
    #np.save("golden_postrotate", rotated.d)
    permuted = row_permute(rotated, c2r_comp_permutes(m, n))
    #np.save("golden_postpermute", permuted.d)
    return permuted

def c2r_transpose(a):
    try:
        return c2r_odd_transpose(a)
    except ValueError:
        try:
            return c2r_po2_transpose(a)
        except ValueError:
            return c2r_comp_transpose(a)
        
def test_c2r(ms, ns):
    for m, n in itertools.product(ms, ns):
        assert(check_row_array(c2r_transpose(make_col_array(m, n))))
            
def invert(y):
    r = [0] * len(y)
    for xi, yi in enumerate(y):
        r[yi] = xi
    return r

def r2c_odd_prerotates(m, n):
    return map(lambda j: (j * (m - mmi(n, m))) % m, range(n))

def r2c_odd_shuffles(m, n):
    result = Array(m, n)
    for i in range(m):
        for j in range(n):
            result[i, j] = ((j * m) + ((i * n) % m)) % n
    return result


def r2c_odd_permutes(m, n):
    return map(lambda i: (i * mmi(n, m)) % m, range(m))

def r2c_odd_transpose(a):
    m, n = a.shape
    if not is_power_of_two(n) or \
            not is_odd(m) or \
            not m > 1:
        raise ValueError("Constraints not satisfied")

    print("Testing R2C Odd Transpose. m, n = (%s, %s)" %(m, n))
    
    prerotated = col_rotate(a, r2c_odd_prerotates(m, n))
    shuffled = row_shuffle(prerotated, r2c_odd_shuffles(m, n))
    permuted = row_permute(shuffled, r2c_odd_permutes(m, n))
    return permuted

def r2c_po2_prerotates(m, n):
    return map(lambda j: j % m, range(n))

def r2c_po2_shuffles(m, n):
    result = Array(m, n)
    def f(i, j):
        return ((j * m *(n + 1))//n) % n - i
    def s(i, j):
        fij = f(i, j)
        if fij >= (m * j) % n:
            return fij % n
        else:
            return (fij + m) % n
    for i in range(m):
        for j in range(n):
            result[i, j] = s(i, j)
    return result

def r2c_po2_permutes(m, n):
    return map(lambda i: (i * (m - 1)) % m, range(m))

def r2c_po2_rotates(m, n):
    return map(lambda j: (m - (j * m)//n) % m, range(n))

def r2c_po2_transpose(a):
    m, n = a.shape
    if not is_power_of_two(n) or \
            not gcd(m, n) == m:
        raise ValueError("Constraints not satisfied")

    print("Testing R2C Po2 Transpose. m, n = (%s, %s)" %(m, n))
        
    prerotated = col_rotate(a, r2c_po2_prerotates(m, n))
    shuffled = row_shuffle(prerotated, r2c_po2_shuffles(m, n))
    permuted = row_permute(shuffled, r2c_po2_permutes(m, n))
    rotated = col_rotate(permuted, r2c_po2_rotates(m, n))
    return rotated

def r2c_comp_prepermutes(m, n):
    return invert(c2r_comp_permutes(m, n))

def r2c_comp_prerotates(m, n):
    return map(lambda j: m - (j % m), range(n))

def r2c_comp_shuffles(m, n):
    c = gcd(m, n)
    def g(i, j):
        return i + ((m * j) % n) + (j * c)//n
    def s(i, j):
        gij = g(i, j)
        if (j * c)//n + i < m:
            return gij % n
        else:
            return (gij - m) % n
    result = Array(m, n)
    for i in range(m):
        for j in range(n):
            result[i, j] = s(i, j)
    return result

def r2c_comp_rotates(m, n):
    c = gcd(m, n)
    return map(lambda j: m - (j * c)//n, range(n))

def r2c_comp_transpose(a):
    #no constraints
    m, n = a.shape
    print("Testing R2C Cmp Transpose. m, n = (%s, %s)" %(m, n))
    
    prepermuted = row_permute(a, r2c_comp_prepermutes(m, n))
    prerotated = col_rotate(prepermuted, r2c_comp_prerotates(m, n))
    shuffled = row_shuffle(prerotated, r2c_comp_shuffles(m, n))
    rotated = col_rotate(shuffled, r2c_comp_rotates(m, n))

    return rotated
        

def r2c_transpose(a):
    try:
        return r2c_odd_transpose(a)
    except ValueError:
        try:
            return r2c_po2_transpose(a)
        except ValueError:
            return r2c_comp_transpose(a)

def test_r2c(ms, ns):
    for m, n in itertools.product(ms, ns):
        assert(check_col_array(r2c_transpose(make_row_array(m, n))))


if __name__ == '__main__':
    test_c2r(range(1, 32), range(1, 32))        
    test_r2c(range(1, 32), range(1, 32))

    print("")
    print("ALL TESTS PASSED")

