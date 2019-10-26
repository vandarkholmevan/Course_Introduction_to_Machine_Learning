import numpy as np
from math import *
from collections import Counter
import sys

def readFile(filename):
    data = []
    inf = open(filename, 'r')
    for i in inf:
        data.append(i.strip('\n').split('\t')[-1])
    return data[1:]

def Entropy(Y):
    '''
    take in array format
    '''
    Y_sum = len(Y)
    Y_cnt = list(Counter(Y).values())
    Y_etp = sum([- i / Y_sum * np.log2(i / Y_sum) for i in Y_cnt])
    return Y_etp

def errRate(Y):
    '''
    take in array format
    '''
    return 1 - max(Counter(Y).values()) / len(Y)

if __name__ == "__main__":
    with open(sys.argv[2], 'w') as ouf:
        Y   = readFile(sys.argv[1])
        etp = Entropy(Y)
        err = errRate(Y)
        ouf.write("entropy: " + str(etp) + "\n")
        ouf.write("error: " + str(err) + "\n")
    ouf.close()
