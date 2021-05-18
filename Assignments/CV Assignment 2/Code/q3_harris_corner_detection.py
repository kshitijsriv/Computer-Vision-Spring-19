import numpy as np
import cv2
import Assignment_1.pad_image as pad
from Assignment_1.gaussian_filter import apply_filter as convolve
import matplotlib.pyplot as plt
import copy

sobel_v = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]
sobel_h = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

threshold = 0.08


def get_gradient(img):
    sobel_vimg = convolve(img, sobel_v, 3)
    sobel_himg = convolve(img, sobel_h, 3)
    print("SHAPE", sobel_himg.shape, sobel_himg.shape)
    return sobel_vimg, sobel_himg


# ref: https://stackoverflow.com/questions/3862225/implementing-a-harris-corner-detector
def harris_corner_detection(ix, iy):
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
    ix2 = cv2.GaussianBlur(ix2, (7, 7), 1.5)
    iy2 = cv2.GaussianBlur(iy2, (7, 7), 1.5)
    ixy = cv2.GaussianBlur(ixy, (7, 7), 1.5)
    c, l = ix.shape

    width = ix.shape[1]
    height = ix.shape[0]

    result = np.zeros((height, width))
    r = copy.deepcopy(result)
    mx = 0
    neighborhood = 3
    for i in range(height):
        for j in range(width):
            m = np.array([
                [ix2[i, j], ixy[i, j]],
                [ixy[i, j], iy2[i, j]]
            ], dtype=np.float64)
            r[i, j] = np.linalg.det(m) - threshold * (np.power(np.trace(m), 2))
            if r[i, j] > mx:
                mx = r[i, j]

    for i in range(height - 1):
        for j in range(width - 1):
            window = np.array(r[(i - int(neighborhood / 2)):(i + int(neighborhood / 2) + 1),
                              (j - int(neighborhood / 2)):(neighborhood + int(neighborhood / 2) + 1)])

            if np.all(window < r[i, j]) and r[i, j] > 0.01 * mx:
                result[i, j] = 1

    pr, pc = np.where(result == 1)
    return np.array(list(zip(pr, pc)))


if __name__ == '__main__':
    input_image = cv2.imread('corner_detection/chess.png')
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    # ROTATE
    # ref: https://www.tutorialkart.com/opencv/python/opencv-python-rotate-image/
    # M = cv2.getRotationMatrix2D((int(gray.shape[1]/2), int(gray.shape[0]/2)), 90, 1.0)
    # gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))

    # COMPRESS
    # ref: https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6
    # gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    # input_image = cv2.resize(input_image, (0, 0), fx=0.5, fy=0.5)

    sobel_y, sobel_x = get_gradient(gray)
    cv2.imwrite('sobel_x.png', sobel_x)
    cv2.imwrite('sobel_y.png', sobel_y)

    # corner_points = harris_corner_detection(sobel_x, sobel_y)
    #
    # for point in corner_points:
    #     input_image[point[0], point[1]] = [0, 0, 255]
    #
    # cv2.imshow('result', input_image)
    # cv2.imwrite('harris_compressed_0.08.png', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
