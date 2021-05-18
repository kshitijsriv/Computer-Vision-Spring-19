import random
import copy
import cv2
import matplotlib.pyplot as plt


def add_snp(image, prctg):
    snp_prob = 0.5

    width = image.shape[1]
    height = image.shape[0]
    snp_count = int(width * height * prctg / 100)

    res_image = copy.deepcopy(image)
    for i in range(snp_count):
        row = random.randint(0, height-1)
        col = random.randint(0, width-1)
        if random.random() < snp_prob:
            res_image[row][col] = 0
        else:
            res_image[row][col] = 255

    return res_image


# input_img = cv2.imread('image_2.png', 0)
# print(input_img)
# # plt.imshow(input_img, cmap='gray')
# # plt.show()
# # cv2.imshow('show', input_img)
#
# res = add_snp(input_img, 10)
# cv2.imshow('show', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
