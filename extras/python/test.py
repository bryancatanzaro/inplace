from shuffle import mmi, gcd, invert

m, n = 14, 7

c = gcd(m, n)
a, b = m/c, n/c

def aug_mmi(x, y):
    return mmi(x, y) or 0

def f(i):
    return (i * n - (i // a)) % m

def candidate(i):
    k = (c - (i % c)) % c
    l = ((c - 1 + i) // c) % a
    return (k * a + (l * aug_mmi(b, a))% a)


fs = map(f, range(m))
gs = invert(fs)
cs = map(candidate, range(m))


print("m, n, c, a, b: %s, %s, %s, %s, %s" % (m, n, c, a, b))
print("mmi(b, m): %s" %(aug_mmi(b, m)))


print(fs)
print(gs)
print(cs)
