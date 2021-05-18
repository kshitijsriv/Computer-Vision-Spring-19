import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from Assignment_1.generate_g_kernel import gen_kernel as g_kernel
from Assignment_1.gaussian_filter import apply_filter
import Assignment_1.scale_image as scale
import Assignment_1.subtract_image as sub

if __name__ == '__main__':
    filter_size = 3
    input_image = cv2.imread('image_3.png', 0)
    print(input_image)
    width = input_image.shape[1]
    height = input_image.shape[0]

    sample = copy.deepcopy(input_image)
    gauss = g_kernel(filter_size, 8.0)
    result_images = []
    laplace_images = []

    result_images.append(input_image)
    for i in range(4):
        res = apply_filter(sample, gauss, filter_size)
        scaled_res = scale.scale_down(res, 2)
        # print("POST SHAPE", scaled_res.shape)
        result_images.append(scaled_res)
        sample = scaled_res

    for i in range(len(result_images) - 1):
        diff = sub.subtract(scale.scale_up(result_images[i], 2 ** i), scale.scale_up(result_images[i + 1], 2 ** (i + 1)))
        laplace_images.append(diff)

    # cv2.imshow('orig', input_image)
    # for i in range(len(result_images)):
    #     cv2.imshow('level' + str(i), result_images[i])

    for i in range(len(laplace_images)):
        cv2.imshow('level' + str(i), scale.scale_down(laplace_images[i], 2 ** i))

    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(141)
    # ax1.imshow(result_images[0], cmap='gray')
    # ax1.set_title('scale factor = 0.5', fontsize=10)
    #
    # ax2 = fig.add_subplot(142)
    # ax2.imshow(result_images[1], cmap='gray')
    # ax2.set_title('scale factor = 0.1', fontsize=10)
    #
    # ax3 = fig.add_subplot(143)
    # ax3.imshow(result_images[2], cmap='gray')
    # ax3.set_title('scale factor = 0.07', fontsize=10)
    #
    # ax4 = fig.add_subplot(144)
    # ax4.imshow(result_images[3], cmap='gray')
    # ax4.set_title('scale factor = 0.05', fontsize=10)
    #
    # fig.add_subplot()
    # # plt.imshow(res_image, cmap='gray')
    # plt.tight_layout()
    # fig = plt.gcf()
    # plt.show()
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()
