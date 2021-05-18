import numpy as np
import cv2
import matplotlib.pyplot as plt

# ref: https://github.com/suhas-nithyanand/Image-Segmentation-using-Region-Growing/blob/master/region_growing.py


def check_threshold(intensity, img_point):
    lw = intensity - 5
    up = intensity + 5
    if lw < img_point < up:
        return True
    else:
        return False


def do_segmentation():
    c = 0
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]

    # intensities = []
    while len(rg_points) > 0:
        if c < 1:
            pt = rg_points.pop(0)
            j = pt[0]
            i = pt[1]
        intensity = img_gray[i][j]
        # intensities.append(intensity)
        # lower_threshold = np.average(intensities) - 100
        # upper_threshold = np.average(intensities) + 100
        for k in range(len(x)):
            if region[i + x[k]][j + y[k]] != 1:
                try:
                    if check_threshold(intensity, img_gray[i + x[k]][j + y[k]]):
                        region[i + x[k]][j + y[k]] = 1
                        new_region_pt = [i + x[k], j + y[k]]
                        if new_region_pt not in rg_points:
                            rg_points.append(new_region_pt)
                    else:
                        rg_points[i + x[k]][j + y[k]] = 0
                except IndexError:
                    continue

        pt = rg_points.pop(0)
        i = pt[0]
        j = pt[1]
        c += 1


if __name__ == '__main__':
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face1.jpg'
    image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face2.jpg'
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face3.jpg'
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q3-faces/face4.jpg'

    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    plt.figure()
    plt.imshow(img_gray, cmap='gray')
    skin_seed = plt.ginput()
    skin_seed = np.array(skin_seed[0], dtype=int).tolist()
    plt.close()
    region = np.zeros((height + 1, width + 1))
    region[skin_seed[1]][skin_seed[0]] = 255
    rg_points = [skin_seed]
    do_segmentation()

    # cv2.imwrite('seeded_segmentation_skin_3.jpg', region)
    plt.figure()
    plt.imshow(region, cmap="gray")
    plt.show()


