import numpy as np
import cv2


def scale_down(img, factor):
    width = img.shape[1]
    height = img.shape[0]
    # print(height, width)

    res = np.zeros([int(height / factor), int(width / factor)], dtype=np.uint8)
    print('pre shape', res.shape)

    for i in range(0, height, factor):
        for j in range(0, width, factor):
            # print(i, j)
            res[int(i/factor)][int(j/factor)] = img[i][j]

    return res


def scale_up(img, factor):
    width = img.shape[1]
    height = img.shape[0]

    res = np.zeros([height * factor, width * factor], dtype=np.uint8)

    for i in range(height * factor):
        for j in range(width * factor):
            res[i][j] = img[int(i/factor)][int(j/factor)]

    return res


# img = cv2.imread('image_3.png', 0)
# img_small = scale_down(img, 2)
# img_big = scale_up(img_small, 2)
#
# cv2.imshow('org', img)
# cv2.imshow('small', img_small)
# cv2.imshow('big', img_big)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
