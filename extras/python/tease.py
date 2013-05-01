from shuffle import *
from transpose import *
import numpy as np

begin = np.loadtxt("begin.txt", dtype=np.int32)
prerotate = np.loadtxt("prerotate.txt", dtype=np.int32)
shuffle = np.loadtxt("shuffle.txt", dtype=np.int32)
postrotate = np.loadtxt("postrotate.txt", dtype=np.int32)
postpermute = np.loadtxt("postpermute.txt", dtype=np.int32)

golden_begin = np.load("golden_begin.npy")
golden_prerotate = np.load("golden_prerotate.npy")
golden_shuffle = np.load("golden_shuffle.npy")
golden_postrotate = np.load("golden_postrotate.npy")
golden_postpermute = np.load("golden_postpermute.npy")

def checks(a, b):
    return (a==b).all()

import pdb
pdb.set_trace()
