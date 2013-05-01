from fractions import gcd

def permutes(m, n):
    c = gcd(m, n)
    return map(lambda i: (i * n - (i * c)//m) % m,
               range(m))

#Finds the cycles of a gather permutation
#returns them as a nested sequence
def find_cycles(p):
    visited = set()
    def find_minimum_unvisited():
        for i in range(len(p)):
            if i not in visited:
                return i
        assert(false)
    result = []
    while len(visited) < len(p):
        start = find_minimum_unvisited()
        pos = start
        visited.add(pos)
        add = False
        while p.index(pos) != start:
            add = True
            pos = p.index(pos)
            visited.add(pos)
        if add:
            result.append(start)
    return result


def test(m, n):
    perms = permutes(m, n)
    cycles = find_cycles(perms)
    print(perms)
    print(cycles)


test(10, 32)
