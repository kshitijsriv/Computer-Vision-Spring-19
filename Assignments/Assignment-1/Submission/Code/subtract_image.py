import numpy as np


def subtract(a, b):
    res = a - b
    res[res > 70] = 0
    return res
