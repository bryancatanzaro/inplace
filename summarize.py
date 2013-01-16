import numpy as np
import re
import sys

import matplotlib.pyplot as plt

def parse_transposes(fn):
    size = re.compile('(\d+) x (\d+)')
    tp = re.compile('Throughput: ([\d\.]+) GB')
    sizes = []
    tps = []
    with open(fn, 'r') as f:
        for l in f:
            s = size.search(l)
            if s:
                sizes.append((int(s.group(1)), int(s.group(2))))
            else:
                t = tp.search(l)
                if t:
                    tps.append(float(t.group(1)))
    return sizes, tps

def summarize(tps):
    return np.histogram(tps, bins=50)

  
if __name__ == '__main__':
    sizes, tps = parse_transposes(sys.argv[1])
    print("Median throughput: %s GB/s" % np.median(tps))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(tps, 50)
    ax.set_xlabel('GB/s')
    ax.set_title(sys.argv[1])
    plt.show()
