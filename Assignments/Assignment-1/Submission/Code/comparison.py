import cv2
import numpy as np
import Assignment_1.average_filter as avg
import Assignment_1.pad_image as pad
import Assignment_1.add_snp_noise as snp
import Assignment_1.median_filter as med
import Assignment_1.gaussian_filter as gaussian
import Assignment_1.generate_g_kernel as g_kernel
import matplotlib.pyplot as plt

# average filter
# avg_img = cv2.imread('image_1.jpg', 0)
# avg_kernel = np.ones((15, 15), np.float32) / 225
# cv_avg_img = cv2.filter2D(avg_img, -1, avg_kernel)
#
# fltr = np.ones((15, 15), np.float32)
# input_image = pad.padding(avg_img, 7)
# own_avg_filter_img = avg.convolve_filter(input_image, fltr)
#
# res_avg = cv2.subtract(cv_avg_img, own_avg_filter_img)

# median filter
# med_img = cv2.imread('image_2.png', 0)
# snp_10_img = snp.add_snp(med_img, 10)
# snp_20_img = snp.add_snp(med_img, 20)
#
# cv_med_10_img = cv2.medianBlur(snp_10_img, 11)
# cv_med_20_img = cv2.medianBlur(snp_20_img, 11)
#
# own_med_filter_10_img = med.apply_filter(pad.padding(snp_10_img, 5), 11)
# own_med_filter_20_img = med.apply_filter(pad.padding(snp_20_img, 5), 11)
#
# res_med_10 = cv2.subtract(cv_med_10_img, own_med_filter_10_img)
# res_med_20 = cv2.subtract(cv_med_20_img, own_med_filter_20_img)

# cv2.imshow('median_10', res_med_10)
# cv2.imshow('median_20', res_med_20)

# gaussian filter
gauss_img = cv2.imread('image_3.png', 0)
gauss_kernel = g_kernel.gen_kernel(15, 8.0)

cv_gauss_img = cv2.GaussianBlur(gauss_img, (15, 15), 5.0)
own_gauss_img = gaussian.apply_filter(pad.padding(gauss_img, 7), gauss_kernel, 15)
print(cv_gauss_img.shape)
print(own_gauss_img.shape)

res_gauss = cv2.subtract(cv_gauss_img, own_gauss_img)
# cv2.imshow('gaussian_own', own_gauss_img)
# cv2.imshow('cv_gaussian', cv_gauss_img)
# cv2.imshow('diff', res_gauss)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
fig = plt.figure()

# average
# ax1 = fig.add_subplot(4, 4, 1)
# ax1.imshow(avg_img, cmap='gray')
# ax1.set_title('Original')
#
# ax2 = fig.add_subplot(4, 4, 2)
# ax2.imshow(cv_avg_img, cmap='gray')
# ax2.set_title('OpenCV')
#
# ax3 = fig.add_subplot(4, 4, 3)
# ax3.imshow(own_avg_filter_img, cmap='gray')
# ax3.set_title('Self-Implementation')
#
# ax4 = fig.add_subplot(4, 4, 4)
# ax4.imshow(res_avg, cmap='gray')
# ax4.set_title('Difference')
#
# # median
# ax5 = fig.add_subplot(4, 4, 5)
# ax5.imshow(snp_10_img, cmap='gray')
# ax5.set_title('Original(SNP 10%)')
#
# ax6 = fig.add_subplot(4, 4, 6)
# ax6.imshow(cv_med_10_img, cmap='gray')
# ax6.set_title('OpenCV')
#
# ax7 = fig.add_subplot(4, 4, 7)
# ax7.imshow(own_med_filter_10_img, cmap='gray')
# ax7.set_title('Self-Implementation')
#
# ax8 = fig.add_subplot(4, 4, 8)
# ax8.imshow(res_med_10, cmap='gray')
# ax8.set_title('Difference')
#
# ax9 = fig.add_subplot(4, 4, 9)
# ax9.imshow(snp_20_img, cmap='gray')
# ax9.set_title('Original(SNP 20%)')
#
# ax10 = fig.add_subplot(4, 4, 10)
# ax10.imshow(cv_med_20_img, cmap='gray')
# ax10.set_title('OpenCV')
#
# ax11 = fig.add_subplot(4, 4, 11)
# ax11.imshow(own_med_filter_20_img, cmap='gray')
# ax11.set_title('Self-Implementation')
#
# ax12 = fig.add_subplot(4, 4, 12)
# ax12.imshow(res_med_20, cmap='gray')
# ax12.set_title('Difference')
#
# # gaussian
# ax13 = fig.add_subplot(4, 4, 13)
# ax13.imshow(gauss_img, cmap='gray')
# ax13.set_title('Original')
#
# ax14 = fig.add_subplot(4, 4, 14)
# ax14.imshow(cv_gauss_img, cmap='gray')
# ax14.set_title('OpenCV')
#
# ax15 = fig.add_subplot(4, 4, 15)
# ax15.imshow(own_gauss_img, cmap='gray')
# ax15.set_title('Self-Implementation')
#
# ax16 = fig.add_subplot(4, 4, 16)
# ax16.imshow(res_gauss, cmap='gray')
# ax16.set_title('Difference')


ax1 = fig.add_subplot(221)
ax1.imshow(gauss_img, cmap='gray')
ax1.set_title('Original')

ax2 = fig.add_subplot(222)
ax2.imshow(cv_gauss_img, cmap='gray')
ax2.set_title('OpenCV')

ax3 = fig.add_subplot(223)
ax3.imshow(own_gauss_img, cmap='gray')
ax3.set_title('Self-Implementation')

ax4 = fig.add_subplot(224)
ax4.imshow(res_gauss, cmap='gray')
ax4.set_title('Difference')

plt.tight_layout()
fig = plt.gcf()
plt.show()
