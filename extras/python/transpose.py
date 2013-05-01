from shuffle import *

def shuffle_transpose(a):
    #This assumes a is in row-major order
    at = c2r_transpose(a)
    #Reassign the shape of the matrix
    #(Performs no data movement)
    m, n = at.shape
    at.shape = n, m
    return at


if __name__ == '__main__':
#a = make_row_array(4,4)
    a = make_row_array(1001,1502)
    at = shuffle_transpose(a)
    ap = shuffle_transpose(at)
    
    assert((a.d.T == at.d).all())
    assert((at.d.T == ap.d).all())
