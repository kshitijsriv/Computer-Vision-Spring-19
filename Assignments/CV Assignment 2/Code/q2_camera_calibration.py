import cv2
import numpy as np
import glob

# reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html  #
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((12 * 12, 3), np.float32)
objp[:, :2] = np.mgrid[0:12, 0:12].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('camera_calibration/*.bmp', recursive=True)

for fname in images:
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (12, 12), None)
    i = 0
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (12, 12), corners2, ret)
        # cv2.imshow('img', img)
        cv2.imwrite('cam_calibrate' + str(i) + '.png', img)
        i += 1
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(rvecs)
# print(tvecs)
# print(dist)
# mean_error = 0
# tot_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     tot_error += error
#
# print("reprojection error: ", tot_error)
# print("mean error: ", tot_error / len(objpoints))
