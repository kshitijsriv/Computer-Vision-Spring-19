# reference: https://www.ijcsmc.com/docs/papers/May2016/V5I5201601.pdf

import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


input_image = cv2.imread('image_3.png', 0)
watermark = cv2.imread('watermark.png', 0)
# scale_factor = 0.7
scale_factor_vals = [0.5, 0.1, 0.07, 0.05]
res_images = []

for scale_factor in scale_factor_vals:
    coeff_image_1 = pywt.dwt2(input_image, 'haar')

    coeff_image_2 = pywt.dwt2(coeff_image_1[0], 'haar')
    coeff_image_3 = pywt.dwt2(coeff_image_2[0], 'haar')

    coeff_wm_1 = pywt.dwt2(watermark, 'haar')
    coeff_wm_2 = pywt.dwt2(coeff_wm_1[0], 'haar')
    coeff_wm_3 = pywt.dwt2(coeff_wm_2[0], 'haar')

    new_ll3 = np.array(coeff_image_3[0]) + scale_factor * np.array(coeff_wm_3[0])
    new_coeffs3 = new_ll3, coeff_image_3[1]

    new_ll2 = pywt.idwt2(new_coeffs3, 'haar')
    new_coeffs2 = new_ll2, coeff_image_2[1]

    new_ll1 = pywt.idwt2(new_coeffs2, 'haar')
    res_coeffs = new_ll1, coeff_image_1[1]

    res_image = pywt.idwt2(res_coeffs, 'haar')
    res_images.append(res_image)

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.imshow(res_images[0], cmap='gray')
ax1.set_title('scale factor = 0.5', fontsize=10)

ax2 = fig.add_subplot(222)
ax2.imshow(res_images[1], cmap='gray')
ax2.set_title('scale factor = 0.1', fontsize=10)

ax3 = fig.add_subplot(223)
ax3.imshow(res_images[2], cmap='gray')
ax3.set_title('scale factor = 0.07', fontsize=10)

ax4 = fig.add_subplot(224)
ax4.imshow(res_images[3], cmap='gray')
ax4.set_title('scale factor = 0.05', fontsize=10)


# plt.imshow(res_image, cmap='gray')
plt.show()
