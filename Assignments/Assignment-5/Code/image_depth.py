import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('left.jpg', 0)
imgR = cv2.imread('right.jpg', 0)

print(imgL.shape, imgR.shape)


stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(imgL, imgR)

cv2.imwrite('disparity_img.png', disparity)

plt.imshow(disparity, 'gray')
plt.show()
