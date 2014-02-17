from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data.npy")

def moving_average(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n

variants, samples = data.shape
ma_data = np.zeros(shape=(variants, samples - 1000 + 1))
for var in range(variants):
    ma_data[var, :] = moving_average(data[var, :], 1000)

plt.plot(ma_data[:,6918:].transpose())
plt.show()
    
# diff_average = moving_average(diff_sq)
# plt.plot(moving_average(data[6,:], 1000))
# plt.show()
import pdb
pdb.set_trace()
switch = ma_data[9] - ma_data[0]
for x in range(len(switch)):
    if (switch[x] > 0):
        print(x)
        break
