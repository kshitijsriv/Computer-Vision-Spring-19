import numpy as np
import math


def gauss_function(x, y, sig):
    a = 1 / (2 * math.pi * sig ** 2)
    exp = math.exp(- (x ** 2 + y ** 2) / (2 * sig ** 2))
    res = a * exp
    return res


def gen_kernel(size, sigma):
    out = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            out[i][j] = gauss_function(i, j, sigma)

    out = out / np.sum(out)
    # print(out)
    return out


# gen_kernel(5, 10)
