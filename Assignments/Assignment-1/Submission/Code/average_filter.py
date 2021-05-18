import matplotlib.pyplot as plt
import cv2
import numpy as np
import Assignment_1.pad_image as pad


def display_image(img):
    window_width = img.shape[1]
    window_height = img.shape[0]

    cv2.namedWindow('img_show', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('img_show', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolve_filter(img, fltr):
    width = img.shape[1]
    height = img.shape[0]

    new_img = np.zeros([height, width])
    # print(width, height)

    filter_size = len(fltr)
    # print("FILTER", filter_size)
    for image_i in range(int(filter_size / 2), height - int(filter_size / 2)):
        for image_j in range(int(filter_size / 2), width - int(filter_size / 2)):
            val = 0
            # print(image_i, image_j)
            # print((image_i + int(filter_size / 2) + 1) - (image_i - int(filter_size / 2)))

            s = np.sum(img[(image_i - int(filter_size / 2)):(image_i + int(filter_size / 2) + 1),
                                        (image_j - int(filter_size / 2)):(image_j + int(filter_size / 2) + 1)])

            new_img[image_i][image_j] = int((s / filter_size ** 2))
            # new_img[image_i][image_j] = np.sum(img[(image_i - int(filter_size / 2)):(image_i + int(filter_size / 2) + 1),
            #                             (image_j - int(filter_size / 2)):(image_j + int(filter_size / 2) + 1)]) / filter_size**2

            # new_img[image_i][image_j] = val/(filter_size ** 2)
    return pad.crop(new_img, int(filter_size / 2))


if __name__ == '__main__':
    input_img = cv2.imread('image_1.jpg', 0)

    # filter_size = input("Enter filter size. E.g. for 5x5 enter 5\n")
    res_images = []

    # print(input_img)
    height = input_img.shape[0]
    width = input_img.shape[1]
    # print(width, height)
    f_sizes = [3, 5, 11, 15]

    fig = plt.figure()

    for fs in f_sizes:
        input_img = pad.padding(input_img, int(fs / 2))
        fltr = np.ones([int(fs), int(fs)])
        res_images.append(convolve_filter(input_img, fltr))

    ax1 = fig.add_subplot(221)
    ax1.imshow(res_images[0], cmap='gray')
    ax1.set_title('3x3')

    ax2 = fig.add_subplot(222)
    ax2.imshow(res_images[1], cmap='gray')
    ax2.set_title('5x5')

    ax3 = fig.add_subplot(223)
    ax3.imshow(res_images[2], cmap='gray')
    ax3.set_title('11x11')

    ax4 = fig.add_subplot(224)
    ax4.imshow(res_images[3], cmap='gray')
    ax4.set_title('15x15')

    plt.tight_layout()
    # plt.title('Average filters')
    fig = plt.gcf()
    plt.show()
    # display_image(res_img)
    # cv2.imshow('show', res_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(res_img, cmap='gray')
    # plt.show()
