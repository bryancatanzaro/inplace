from fractions import gcd
from shuffle import mmi
import random

def gather_permutes(m, n):
    c = gcd(m, n)
    return map(lambda i: (i * n - (i * c)//m) % m,
               range(m))

def invert(x):
    def find(xi):
        r = x.index(xi)
        return r
    return map(lambda xi: find(xi), range(len(x)))

def scatter_permutes(m, n):
    c = gcd(m, n)
    a = m / c
    b = n / c
    q = mmi(b, a) or 0
    def el(i):
        k = ((c - 1) * i) % c
        l = ((c - 1 + i) // c)
        return (k * a + (l * q) % a) 
    return map(el, range(m))


def gather_permute(d, g):
    return map(lambda gi: d[gi], g)

def scatter_permute(d, s):
    r = [0] * len(d)
    for di, si in zip(d, s):
        r[si] = di
    return r

def scatter_cycles(s):
    m = len(s)
    unvisited = dict(zip(range(m), s))
    heads = []
    while(unvisited):
        idx, dest = unvisited.popitem()
        if idx != dest:
            heads.append(idx)
            start = idx
            while dest != start:
                idx = dest
                dest = unvisited.pop(idx)
    return heads
        


def test(m, n):
    print("(m, n): (%s, %s)" %(m, n))

    d = range(m)
    
    c = gcd(m, n)
    a = m / c
    b = n / c


    x = gather_permutes(m, n)
    z = scatter_permutes(m, n)

    gathered = gather_permute(d, x)
    scattered = scatter_permute(d, z)

    assert(gathered == scattered)

if __name__ == '__main__':
    # for i in range(1000):
    #     m = random.randint(1, 100000)
    #     n = random.randint(1, 100000)
    #     test(m, n)
    m = 6
    n = 32
    z = scatter_permutes(m, n)
    print(z)
    heads = scatter_cycles(z)
    print(heads)
