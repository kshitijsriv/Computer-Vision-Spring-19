import numpy as np


def padding(img, pad):
    width = img.shape[1]
    height = img.shape[0]
    res = np.zeros([height + 2 * pad, width + 2 * pad], dtype=np.uint8)
    # print(res.shape)
    for i in range(height):
        for j in range(width):
            res[i + pad][j + pad] = img[i][j]
    return res


def crop(img, pad):
    width = img.shape[1] - 2 * pad
    height = img.shape[0] - 2 * pad
    res = np.zeros([height, width], dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            res[i][j] = img[i + pad][j + pad]

    return res

#
# a = np.arange(100)
# a = a.reshape(10, 10)
# padded = padding(a, 1)
# print(padded)
# print(crop(padded, 1))
