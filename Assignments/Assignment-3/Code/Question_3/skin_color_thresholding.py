import cv2
import numpy as np
from copy import deepcopy

# ref: https://github.com/WillBrennan/SkinDetector/tree/master/skin_detector


class SkinThresholding:
    def __init__(self, rgb, ycrcb, hsv):
        self.img_rgb = rgb
        self.img_ycrcb = ycrcb
        self.img_hsv = hsv
        self.height = rgb.shape[0]
        self.width = rgb.shape[1]
        self.thresh = 0.5

        print(type(self.img_rgb[0][0][1]))
        print("RGB SHAPE", rgb.shape, ycrcb.shape, hsv.shape)

    def rgb_mask(self, rgb):
        lower_thresh = np.array([45, 52, 108], dtype=np.uint8)
        upper_thresh = np.array([255, 255, 255], dtype=np.uint8)

        mask_a = cv2.inRange(rgb, lower_thresh, upper_thresh)
        mask_b = 255 * ((rgb[:, :, 2] - rgb[:, :, 1]) / 20)
        mask_c = 255 * ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) / 20)
        mask_d = np.bitwise_and(np.uint64(mask_a), np.uint64(mask_b))
        mask_rgb = np.bitwise_and(np.uint64(mask_c), np.uint64(mask_d))

        mask_rgb[mask_rgb < 128] = 0
        mask_rgb[mask_rgb >= 128] = 1

        return mask_rgb.astype(float)

    def ycrcb_mask(self, ycrcb):
        lower_thresh = np.array([90, 100, 130], dtype=np.uint8)
        upper_thresh = np.array([230, 120, 180], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_thresh, upper_thresh)

        mask_ycrcb[mask_ycrcb < 128] = 0
        mask_ycrcb[mask_ycrcb >= 128] = 1

        return mask_ycrcb.astype(float)

    def hsv_mask(self, hsv):
        lower_thresh = np.array([0, 50, 0], dtype=np.uint8)
        upper_thresh = np.array([120, 150, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_thresh, upper_thresh)

        mask_hsv[mask_hsv < 128] = 0
        mask_hsv[mask_hsv >= 128] = 1
        return mask_hsv.astype(float)

    def get_res(self):
        return self.img_rgb

    def sk_threshold(self):
        mask_hsv = self.hsv_mask(self.img_hsv)
        mask_rgb = self.rgb_mask(self.img_rgb)
        mask_ycrcb = self.ycrcb_mask(self.img_ycrcb)

        n_masks = 3.0
        mask = (mask_hsv + mask_rgb + mask_ycrcb) / n_masks

        mask[mask < self.thresh] = 0.0
        mask[mask >= self.thresh] = 255.0

        mask = mask.astype(np.uint8)
        return mask


if __name__ == '__main__':
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face1.jpg'
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face2.jpg'
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face3.jpg'
    image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face4.jpg'
    rgb = cv2.imread(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.medianBlur(rgb, ksize=3)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCR_CB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
    ycrcb = cv2.resize(ycrcb, (0, 0), fx=0.5, fy=0.5)
    hsv = cv2.resize(hsv, (0, 0), fx=0.5, fy=0.5)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    sk_thres = SkinThresholding(deepcopy(rgb), deepcopy(ycrcb), deepcopy(hsv))
    res = sk_thres.sk_threshold()
    cv2.imwrite('skin_4.jpg', res)
    cv2.imshow('orig', rgb)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
