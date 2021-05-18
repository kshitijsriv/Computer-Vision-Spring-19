import matplotlib.pyplot as plt
import cv2
import numpy as np
from Assignment_1.generate_g_kernel import gen_kernel as g_kernel
import copy
import Assignment_1.pad_image as pad


def apply_filter(img, kernel, filter_size):
    res_image = copy.deepcopy(img)
    width = img.shape[1]
    height = img.shape[0]

    # print("FILTER", filter_size)
    for image_i in range(int(filter_size / 2), height - int(filter_size / 2)):
        for image_j in range(int(filter_size / 2), width - int(filter_size / 2)):
            val = 0
            # print(image_i, image_j)
            # print((image_i + int(filter_size / 2) + 1) - (image_i - int(filter_size / 2)))

            s = img[(image_i - int(filter_size / 2)):(image_i + int(filter_size / 2) + 1),
                (image_j - int(filter_size / 2)):(image_j + int(filter_size / 2) + 1)]

            for i in range(len(s)):
                for j in range(len(s)):
                    val += kernel[i][j] * s[i][j]
            res_image[image_i][image_j] = val

    return pad.crop(res_image, int(filter_size / 2))
    # return res_image


if __name__ == '__main__':
    # filter_size = input("Enter filter size\n")
    # filter_size = int(filter_size)
    # filter_size = 3
    input_image = cv2.imread('image_3.png', 0)

    width = input_image.shape[1]
    height = input_image.shape[0]

    f_sizes = [3, 5, 11, 15]
    res = []

    # for filter_size in f_sizes:
    #     print(filter_size)
    #     img = pad.padding(img=copy.deepcopy(input_image), pad=int(filter_size / 2))
    #     gauss = g_kernel(filter_size, sigma=5.0)
    #     res.append(apply_filter(input_image, gauss, filter_size))
    #
    sig = [1.0, 2.0, 4.0, 8.0, 16.0]
    #
    fltr_size = 11
    img = pad.padding(img=copy.deepcopy(input_image), pad=int(fltr_size / 2))

    for sigma in sig:
        gauss = g_kernel(fltr_size, sigma=sigma)
        res.append(apply_filter(img, gauss, fltr_size))

    fig = plt.figure()

    ax1 = fig.add_subplot(231)
    ax1.imshow(input_image, cmap='gray')
    ax1.set_title('original')

    ax2 = fig.add_subplot(232)
    ax2.imshow(res[0], cmap='gray')
    ax2.set_title('sigma = 1.0')
    # ax2.set_title('3x3')

    ax3 = fig.add_subplot(233)
    ax3.imshow(res[1], cmap='gray')
    ax3.set_title('sigma = 2.0')
    # ax3.set_title('5x5')

    ax4 = fig.add_subplot(234)
    ax4.imshow(res[2], cmap='gray')
    ax4.set_title('sigma = 4.0')
    # ax4.set_title('11x11')

    ax5 = fig.add_subplot(235)
    ax5.imshow(res[3], cmap='gray')
    ax5.set_title('sigma = 8.0')
    # ax5.set_title('15x15')

    ax6 = fig.add_subplot(236)
    ax6.imshow(res[4], cmap='gray')
    ax6.set_title('sigma = 16.0')

    # plt.imshow(res, cmap='gray')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
