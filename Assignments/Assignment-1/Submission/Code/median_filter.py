import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
from Assignment_1.add_snp_noise import add_snp as snp
import Assignment_1.pad_image as pad


def apply_filter(img, filter_size):
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

            s = np.array(s)
            s = s.flatten()
            s = sorted(s)

            res_image[image_i][image_j] = s[int(filter_size ** 2 / 2)]
            # new_img[image_i][image_j] = np.sum(img[(image_i - int(filter_size / 2)):(image_i + int(filter_size / 2) + 1),
            #                             (image_j - int(filter_size / 2)):(image_j + int(filter_size / 2) + 1)]) / filter_size**2

            # new_img[image_i][image_j] = val/(filter_size ** 2)

    return pad.crop(res_image, int(filter_size / 2))


if __name__ == '__main__':
    input_img = cv2.imread('image_2.png', 0)
    # filter_size = input("Enter filter size. E.g. for 5x5 enter 5\n")
    # filter_size = int(filter_size)

    res_images = []
    f_sizes = [3, 5, 11]

    # print(input_img)
    height = input_img.shape[0]
    width = input_img.shape[1]

    snp_img = snp(input_img, 10)
    for fs in f_sizes:
        filter_size = fs
        s_image = pad.padding(snp_img, int(filter_size / 2))
        res_images.append(apply_filter(s_image, filter_size))

    fig = plt.figure()

    ax1 = fig.add_subplot(241)
    ax1.imshow(snp_img, cmap='gray')
    ax1.set_title('SNP 10%')

    ax2 = fig.add_subplot(242)
    ax2.imshow(res_images[0], cmap='gray')
    ax2.set_title('3x3')

    ax3 = fig.add_subplot(243)
    ax3.imshow(res_images[1], cmap='gray')
    ax3.set_title('5x5')

    ax4 = fig.add_subplot(244)
    ax4.imshow(res_images[2], cmap='gray')
    ax4.set_title('11x11')

    res_images = []

    snp_img_20 = snp(input_img, 20)
    for fs in f_sizes:
        filter_size = fs
        s_image = pad.padding(snp_img_20, int(filter_size / 2))
        res_images.append(apply_filter(s_image, filter_size))

    ax5 = fig.add_subplot(245)
    ax5.imshow(snp_img_20, cmap='gray')
    ax5.set_title('SNP 20%')

    ax6 = fig.add_subplot(246)
    ax6.imshow(res_images[0], cmap='gray')
    ax6.set_title('3x3')

    ax7 = fig.add_subplot(247)
    ax7.imshow(res_images[1], cmap='gray')
    ax7.set_title('5x5')

    ax8 = fig.add_subplot(248)
    ax8.imshow(res_images[2], cmap='gray')
    ax8.set_title('11x11')

    plt.tight_layout()
    fig = plt.gcf()

    plt.show()
    # cv2.imread(res)
    # cv2.imshow('show', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
