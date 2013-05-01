import random


def gcd(a, b):
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b:
        quotient = a / b
        new_b = a % b
        a = b
        b = new_b
        new_x = last_x - quotient * x
        last_x = x
        x = new_x
        new_y = last_y - quotient * y
        last_y = y
        y = new_y
    return a




def inplace_rotate(d, r):
    if r == 0:
        return
    m = len(d)
    c = gcd(r, m)
    inc = m - r
    
    def update_pos(pos):
        next_pos = pos + inc
        if next_pos >= m:
            return next_pos - m
        else:
            return next_pos
    
    for b in range(c):
        pos = b
        prior = d[pos]
        next_pos = update_pos(pos)
        while next_pos >= c:
            temp = d[next_pos]
            pos = next_pos
            d[pos] = prior
            prior = temp
            next_pos = update_pos(pos)
        d[next_pos] = prior


def rotate(d, r):
    m = len(d)
    result = [0] * m
    for i in range(m):
        result[i] = d[(i+r)%m]
    return result



for t in range(100000):
    m = random.randint(1,10000)
    r = random.randint(0, m-1)
    a = range(0, m)
    o = rotate(a, r)
    inplace_rotate(a, r)
    equal = a == o
    if not equal:
        print("Failed test #%s: (%s, %s)" % (t, m, r))
    assert(equal)
    print("Passed test #%s: (%s, %s)" % (t, m, r))
