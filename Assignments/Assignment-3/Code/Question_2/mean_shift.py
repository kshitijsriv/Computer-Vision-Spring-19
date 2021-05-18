import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from copy import deepcopy

# image_path = '/home/kshitij/PycharmProjects/Computer_Vision/Assignment_3/Question_2/iceCream1.jpg'
# image_path = '/home/kshitij/PycharmProjects/Computer_Vision/Assignment_3/Question_2/iceCream2.jpg'
image_path = '/home/kshitij/PycharmProjects/Computer_Vision/Assignment_3/Question_2/iceCream3.jpg'

image = cv2.imread(image_path)
image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
orig_img = deepcopy(image)
sh = image.shape
flat_image = image.reshape((image.shape[0] * image.shape[1], 3))
bandwidth2 = estimate_bandwidth(flat_image, quantile=.04, n_samples=1000)
ms = MeanShift(bandwidth2, bin_seeding=True)
ms.fit(flat_image)
labels = ms.labels_

for i in range(len(labels)):
    label = labels[i]
    flat_image[i] = ms.cluster_centers_[label]

print("DONE CLUSTERING")

res = flat_image.reshape(sh)

# cv2.imshow('orig', orig_img)
# cv2.imshow('res', res)
cv2.imwrite('clustered_iceCream3.jpg', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
