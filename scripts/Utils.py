import numpy as np
def gcd(a, b):
    if b > a:
        tmp = a
        a = b
        b = tmp
    while b > 0:
        c = b
        b = a % b
        a = c
    return a


def lcm_pr(a, b):
    return a * b // gcd(a, b)

def diag_block_mat_slicing(L):
    shp = L[0].shape
    N = len(L)
    r = range(N)
    out = np.zeros((N,shp[0],N,shp[1]),dtype=int)
    out[r,:,r,:] = L
    return out.reshape(np.asarray(shp)*N)