# ref: https://github.com/hughesj919/HomographyEstimation

import numpy as np
from matplotlib import pyplot as plt
import cv2


def findHomography(point1, point2):

    H_list = []

    a2 = [0, 0, 0, -point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2),
          point2.item(1) * point1.item(0), point2.item(1) * point1.item(1), point2.item(1) * point1.item(2)]
    a1 = [-point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2), 0, 0, 0,
          point2.item(0) * point1.item(0), point2.item(0) * point1.item(1), point2.item(0) * point1.item(2)]
    H_list.append(a1)
    H_list.append(a2)

    matrixA = np.matrix(H_list)

    # svd composition
    u, s, v = np.linalg.svd(matrixA)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    h = (1 / h.item(8)) * h
    return h


def findDetectionPoints(image1, image2):
    ratio = 0.36
    # ratio = 0.5
    sift = cv2.xfeatures2d.SIFT_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    cv2.imshow('img_keypoints', cv2.drawKeypoints(img_, kp1, None))

    keypointImage1 = cv2.drawKeypoints(image1, kp1, None, (255, 0, 0))
    keypointImage2 = cv2.drawKeypoints(image2, kp2, None, (255, 0, 0))

    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    matchedImage = cv2.drawMatches(keypointImage1, kp1, keypointImage2, kp2, goodMatches, None, (0, 255, 0), flags=2)

    points1 = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in goodMatches])
    # print(points1)
    # print("*********")
    # print(points2)
    # HMatrix, mask = find_homography(points1, points2, cv2.RANSAC, reprojectionThreshold)
    HMatrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, reprojectionThreshold)
    print("HM", HMatrix)
    return points1, points2, matchedImage
    # print(findHomography(points1, points2))


imageB = cv2.imread('collage.jpg')
imageA = cv2.imread('test2.jpeg')
# imageA = cv2.imread('test1.jpeg')


points1, points2, matchedImage = findDetectionPoints(imageA, imageB)

HMatrix, mask = cv2.findHomography(points1, points2)
# print(HMatrix)
height, width = imageA.shape[:2]
points = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)

dataPoints = cv2.perspectiveTransform(points, HMatrix)
dataPoints += (width, 0)

# print(dataPoints)
finalImage = cv2.polylines(matchedImage, [np.int32(dataPoints)], True, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imwrite('matched_test_2.png', finalImage)
cv2.imshow("final image", finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
