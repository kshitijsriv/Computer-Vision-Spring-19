import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import Assignment_1.add_snp_noise as snp


img = cv2.imread('image_3.png', 0)
print(img.shape)
noise_img = snp.add_snp(img, 10)

# reference : https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
ddepth = cv2.CV_16S
kernel_size = 3
denoised = cv2.GaussianBlur(img, (3, 3), 8)
dst = cv2.Laplacian(denoised, ddepth, kernel_size)
abs_dst = cv2.convertScaleAbs(dst)

temp = cv2.add(img, noise_img)
final = cv2.add(temp, abs_dst)

coeffs = pywt.dwt2(final, 'haar')

cA, (cH, cV, cD) = coeffs
d = 1, (2, 3, 4)
print(d)

thres = 3 * np.std(cA)
a = coeffs[0]
# a = pywt.threshold(coeffs[0], thres, 'soft')
b = pywt.threshold(coeffs[1][0], thres, 'soft', np.mean(cH))
c = pywt.threshold(coeffs[1][1], thres, 'soft', np.mean(cV))
d = pywt.threshold(coeffs[1][2], thres, 'soft', np.mean(cD))

coeff = a, (b, c, d)

# plt.imshow(a, cmap='gray')
# plt.show()

# print("#", len(coeffs[1][1]))
#
fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.imshow(cA, cmap='gray')
ax1.set_title('LL')

ax2 = fig.add_subplot(222)
ax2.imshow(cH, cmap='gray')
ax2.set_title('LH')

ax3 = fig.add_subplot(223)
ax3.imshow(cV, cmap='gray')
ax3.set_title('HL')

ax4 = fig.add_subplot(224)
ax4.imshow(cD, cmap='gray')
ax4.set_title('HH')

plt.tight_layout()
fig = plt.gcf()
plt.show()
#
# coeffs_2 = pywt.dwt2(cA, 'haar')
#
# cA_2, (cH_2, cV_2, cD_2) = coeffs_2

fig = plt.figure()

# ax1 = fig.add_subplot(221)
# ax1.imshow(cA_2, cmap='gray')
# ax1.set_title('LL')
#
# ax2 = fig.add_subplot(222)
# ax2.imshow(cH_2, cmap='gray')
# ax2.set_title('LH')
#
# ax3 = fig.add_subplot(223)
# ax3.imshow(cV_2, cmap='gray')
# ax3.set_title('HL')
#
# ax4 = fig.add_subplot(224)
# ax4.imshow(cD_2, cmap='gray')
# ax4.set_title('HH')


res = pywt.idwt2(coeff, 'haar')

ax1 = fig.add_subplot(121)
ax1.imshow(final, cmap='gray')
ax1.set_title('Noisy image')

ax2 = fig.add_subplot(122)
ax2.imshow(res, cmap='gray')
ax2.set_title('Result')
#
plt.tight_layout()
fig = plt.gcf()
plt.show()
