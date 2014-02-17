import subprocess
import itertools
import numpy as np
import copy

def smem_variants(sm):
    for blks in range(8, 0, -1):
        yield "-DSMEM -DSM=%s -DBLKS=%s" % (sm, blks)

def reg_variants(sm):
    for wpt in [16, 18, 30, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
        yield "-DREG -DSM=%s -DWPT=%s" % (sm, wpt)
        
sm = "sm_35"
#variants = list(itertools.chain(smem_variants(sm), reg_variants(sm)))
variants = list(smem_variants(sm))
#variants = list(reg_variants(sm))
#variants = ["-DSMEM -DSM=sm_35 -DBLKS=8"]
#variants = ["-DSMEM -DSM=sm_35, -DBLKS=1", "-DREG -DSM=sm_35 -DWPT=6",
#            "-DREG -DSM=sm_35 -DWPT=7"]

#variants = list(reg_variants(sm))
n_variants = len(variants)
print("n_variants: %s" % n_variants)
data = np.zeros(shape=(n_variants, 29439))
for idx, variant in enumerate(variants):
    cmd_line = "nvcc -I../../inplace -L../../build -linplace -Xptxas -v -arch=%s %s tuner.cu -o tuner" % (sm, variant)
    output = subprocess.check_output(cmd_line, shell=True)
    d = subprocess.check_output("./tuner", shell=True)
    split_data = d.rstrip().split(" ")
    np_data = np.array(split_data)
    data[idx, :] = np_data
    np.save('data', data)
