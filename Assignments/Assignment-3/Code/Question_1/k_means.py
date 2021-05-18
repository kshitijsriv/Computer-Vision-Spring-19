import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
import random
from mpl_toolkits.mplot3d import Axes3D


class KMeans:
    def __init__(self, img, k, iteration_flag=False, centroid_invariant_flag=True, n=None, five_dims_flag=False, height=None, width=None):
        self.data = img
        self.k = k
        self.centroids = []
        self.clusters = defaultdict(list)
        self.cluster_pixel_pos = defaultdict(list)
        self.iteration_flag = iteration_flag
        self.centroid_invariant_flag = centroid_invariant_flag
        self.n = n
        self.five_dims_flag = five_dims_flag
        self.height = height
        self.width = width

    def get_initial_seeds(self):
        for i in range(self.k):
            r_cen = np.random.randint(np.min(self.data[:, 0]), np.max(self.data[:, 0]))
            g_cen = np.random.randint(np.min(self.data[:, 1]), np.max(self.data[:, 1]))
            b_cen = np.random.randint(np.min(self.data[:, 2]), np.max(self.data[:, 2]))
            cen = [r_cen, g_cen, b_cen]
            if self.five_dims_flag:
                x_cen = random.randint(0, self.width)
                y_cen = random.randint(0, self.height)
                cen.append(x_cen)
                cen.append(y_cen)
            self.centroids.append(cen)
        # print("WH", self.width, self.height)
        # print(self.centroids)

    def calc_euclidean_dist(self, point):
        distances = []
        for centroid in self.centroids:
            eu_dist = np.linalg.norm(centroid - point)
            distances.append(eu_dist)
        return distances

    def stopping_criteria(self, old_centroids=None, new_centroids=None):
        # print(old_centroids, new_centroids)
        if np.array_equal(old_centroids, new_centroids):
            return True

    def calc_centroids(self, cluster_dict):
        self.centroids.clear()
        for key in cluster_dict.keys():
            cluster = cluster_dict[key]
            self.centroids.append(np.mean(cluster, axis=0))

    def fill_mean(self):
        for key in self.clusters:
            cluster = np.array(self.clusters[key])
            # print(type(cluster))
            mean = np.mean(cluster, axis=0)
            cluster[:, :] = mean.astype(int)
            self.clusters[key] = cluster

    def gen_image(self):
        result = np.zeros((len(self.data), 3))
        for key in self.cluster_pixel_pos:
            cluster = self.clusters[key]
            pos = self.cluster_pixel_pos[key]
            print("LENGTH", len(cluster), len(pos))
            for i in range(len(cluster)):
                result[pos[i]] = cluster[i][:3]
        # print(result.reshape(orig_shape).shape)
        # plt.imshow(result.reshape(orig_shape))
        # plt.show()
        return result

    def clustering(self):
        self.get_initial_seeds()
        if not self.five_dims_flag:
            old_centroids = np.zeros((self.k, 3))
        else:
            old_centroids = np.zeros((self.k, 5))
        if self.centroid_invariant_flag:
            while not self.stopping_criteria(old_centroids, self.centroids):
                old_centroids = deepcopy(self.centroids)
                clusters = defaultdict(list)
                cluster_pixel_pos = defaultdict(list)
                for i in range(len(self.data)):
                    pixel = self.data[i]
                    clstr = np.argmin(self.calc_euclidean_dist(pixel))
                    clusters[clstr].append(pixel)
                    cluster_pixel_pos[clstr].append(i)

                self.clusters = deepcopy(clusters)
                self.cluster_pixel_pos = deepcopy(cluster_pixel_pos)
                self.calc_centroids(self.clusters)
        print("CLUSTERING DONE !!")
        print(self.centroids)
        self.fill_mean()


def unfold_image_with_dim(img):
    height = img.shape[0]
    width = img.shape[1]
    output_img = []
    for i in range(height):
        for j in range(width):
            val = img[i][j].tolist() + [width, height]
            output_img.append(val)
    output_img = np.array(output_img)
    # print(output_img.shape)
    return np.array(output_img)


def scatter_plot(img):
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    # print(len(r))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, g, b)
    plt.show()


if __name__ == '__main__':
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q1-images/2apples.jpg'
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q1-images/variableObjects.jpeg'
    # image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q1-images/7apples.jpg'
    image = '/home/kshitij/Documents/Computer Vision/Assignments/Assignment-3/Q1-images/2or4objects.jpg'
    img = cv2.imread(image, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    orig_shape = img.shape
    print(img.shape)
    width = img.shape[1]
    height = img.shape[0]

    # with X and Y
    img = unfold_image_with_dim(img)
    print(img.shape)

    # Only 3D color space
    # img = img.reshape((img.shape[0] * img.shape[1], 3))

    k_means = KMeans(img, k=7, five_dims_flag=True, height=height, width=width)
    k_means.clustering()
    clustered_img = k_means.gen_image()
    res = clustered_img.reshape(orig_shape)
    print(np.unique(res))
    # cv2.imwrite('plots/5_dims_2or4objects_k_10.jpg', res)

    scatter_plot(res)
    # print(np.unique(res))
    # print('UNIQUE', res)

    # cv2.imshow('res', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(res, interpolation='nearest')
    # plt.show()
