import math
import os

from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors, plot_matches
import matplotlib.pyplot as plt
from skimage.measure import ransac
import numpy as np

from skimage.transform import FundamentalMatrixTransform


import cv2


def draw_square(img, x, y):
    for i in range(10):
        for j in range(10):
            if img.shape[1] > y-j >= 0 and 0 <= x-i < img.shape[0]:
                img[x-i][y-j] = (255,0,0)
    return img


def merge_images(img1, img2):
    new_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2]))
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    return new_img


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_epipolar_points(img1, F, inliers_img2):
    L = np.cross(np.array([1, 1, 1]), np.array([1, img1.shape[0], 1]))
    R = np.cross(np.array([img1.shape[1], 1, 1]), np.array([img1.shape[1], img1.shape[0], 1]))
    lines1 = []
    for x, y in inliers_img2:
        r = np.array([x, y, 1])
        e = F.dot(r)
        w = np.cross(e, L)
        q = np.cross(e, R)

        X = np.array([w[0]/w[2], q[0]/q[2]])
        Y = np.array([w[1]/w[2], q[1]/q[2]])
        lines1.append((X, Y))
    ep1 = line_intersection(lines1[0], lines1[1])
    return ep1, img1


def draw_epipolar_lines(img1, img2, ep1, ep2, points):
    for point in points:
        cv2.line(img1, (int(ep1[0]), int(ep1[1])), (int(point[0]), int(point[1])),(255,0,0),5)
        cv2.line(img2, (int(ep2[0]), int(ep2[1])), (int(point[0]), int(point[1])),(255,0,0),5)
    cv2.imwrite('res08.png', merge_images(img1, img2))


def get_epipolar_points_svd(F):
    [U, S, V] = np.linalg.svd(F)
    V = [V[0,2], V[1,2], V[2,2]]
    U = [U[0,2], U[1,2], U[2,2]]
    epipol1 = (V[0]/V[2], V[1]/V[2])
    epipol2 = (U[0]/U[2], U[1]/U[2])
    return epipol1, epipol2


def draw_point(img, x, y, color, size=20):
    for i in range(size):
        for j in range(size):
            if x-i >= 0 and x-j >= 0:
                img[x-i][y-j] = color
    return img


def draw_intrest_points(img1, img2, keypoints1, keypoints2):
    for keypoint in keypoints1:
        x, y = keypoint
        img1 = draw_point(img1, int(x), int(y), (0,255,0))
    for keypoint in keypoints2:
        x, y = keypoint
        img2 = draw_point(img2, int(x), int(y), (0,255,0))

    cv2.imwrite('res01.png', merge_images(img1, img2))
    return img1, img2


def draw_matches(img1, img2, matches, keypoints1, keypoints2, img_name, color, size=20):
    for mat in matches:
        x1, y1 = keypoints1[mat[0]]
        img1 = draw_point(img1, int(x1), int(y1), color, size)
        x2, y2 = keypoints2[mat[1]]
        img2 = draw_point(img2, int(x2), int(y2), color, size)
    cv2.imwrite('%s.png' % img_name, merge_images(img1, img2))
    return img1, img2


def draw_inliers(img1, img2, inliers, keypoints1, keypoints2, img_name):
    if not os.path.exists('./%s' % img_name):
        os.makedirs('./%s' % img_name)
    for index, mat in enumerate(inliers):
        fig, ax = plt.subplots()
        plot_matches(ax, img1, img2, keypoints1, keypoints2, np.array([[mat[0], mat[1]]]), only_matches=True)
        ax.axis('off')
        fig.savefig('./%s/%s%d.png' % (img_name, img_name, index), dpi=100, pad_inches=0, bbox_inches='tight')


def check_F(F, matches, keypoints1, keypoints2):
    errors = []
    for mat in matches:
        x1, y1 = keypoints1[mat[0]]
        x2, y2 = keypoints2[mat[1]]
        first_arr = np.array([x1, y1, 1])
        second_arr = np.array([x2, y2, 1])
        # err = first_arr.dot(F).dot(second_arr)
        err = second_arr.dot(F).dot(first_arr)
        errors.append(abs(err))
    # plt.plot(range(1, len(errors)+1), errors, 'ro')
    # plt.axis([0, len(errors) + 1, 0, max(errors)])
    # plt.show()
    return errors


def calculate_F(keypoints, matches):
    model_ransac, inliers = ransac((keypoints[0][matches[:, 0]][:, ::-1], keypoints[1][matches[:, 1]][:, ::-1]),
                                   FundamentalMatrixTransform, min_samples=8, residual_threshold=0.5, max_trials=1000)
    F = model_ransac.__dict__['params']
    return F, inliers


def get_var(items):
    mean = sum(items) / len(items)
    var = 0.0
    for item in items:
        var += ((item - mean)**2)
    var /= len(items)
    return math.sqrt(var)


def run():
    img1 = cv2.imread('01.JPG')
    new_img1 = cv2.imread('01.JPG')
    img2 = cv2.imread('02.JPG')
    new_img2 = cv2.imread('02.JPG')

    images12 = [img1, img2]
    images21 = [img2, img1]
    orb = ORB(n_keypoints=800, fast_threshold=0.05)
    keypoints12 = []
    descriptors12 = []
    for image in images12:
        orb.detect_and_extract(rgb2gray(image))
        keypoints12.append(orb.keypoints)
        descriptors12.append(orb.descriptors)
    keypoints21= []
    descriptors21 = []
    for image in images21:
        orb.detect_and_extract(rgb2gray(image))
        keypoints21.append(orb.keypoints)
        descriptors21.append(orb.descriptors)
    new_img1, new_img2 = draw_intrest_points(new_img1, new_img2, keypoints12[0], keypoints12[1])
    print 'saved res01'

    matches12 = match_descriptors(descriptors12[0], descriptors12[1], cross_check=True)
    matches21 = match_descriptors(descriptors21[0], descriptors21[1], cross_check=True)
    new_img1, new_img2 = draw_matches(new_img1, new_img2, matches12, keypoints12[0], keypoints12[1], 'res02', (255,0,0))
    print 'saved res02'

    fig, ax = plt.subplots()
    plot_matches(ax, new_img1, new_img2, keypoints12[0], keypoints12[1], matches12, keypoints_color='b', matches_color='b', only_matches=True)
    ax.axis('off')
    fig.savefig('./res03.png', dpi=100, pad_inches=0, bbox_inches='tight')
    print 'saved res03'

    fig, ax = plt.subplots()
    plot_matches(ax, img1, img2, keypoints12[0], keypoints12[1], matches12[:20], keypoints_color='b', matches_color='b', only_matches=True)
    ax.axis('off')
    fig.savefig('./res04.png', dpi=100, pad_inches=0, bbox_inches='tight')
    print 'saved res04'

    F12, inliers12 = calculate_F(keypoints12, matches12)
    F21, inliers21 = calculate_F(keypoints21, matches21)
    print 'Fandamental matrix is ', F12

    new_img1, new_img2 = draw_matches(new_img1, new_img2, matches12[inliers12], keypoints12[0], keypoints12[1], 'res05', (0,0,255), size=40)
    print 'saved res05'
    draw_inliers(img1, img2, matches12[inliers12][:5], keypoints12[0], keypoints12[1], 'inlier')
    print 'saved inliers'
    draw_inliers(img1, img2, matches12[np.logical_not(inliers12)][:5], keypoints12[0], keypoints12[1], 'outlier')
    print 'saved outliers'

    inliers_img1 = []
    for first_idx, second_idx in matches21[inliers21]:
        inliers_img1.append(keypoints21[0][first_idx])
    inliers_img2 = []
    for first_idx, second_idx in matches12[inliers12]:
        inliers_img2.append(keypoints12[1][second_idx])
    ep1, ep_img1 = get_epipolar_points(img1, F12, inliers_img2)
    ep2, ep_img2 = get_epipolar_points(img2, F21, inliers_img1)
    ep11, ep22 = get_epipolar_points_svd(F12)
    img1 = cv2.imread('01.JPG')
    img2 = cv2.imread('02.JPG')
    draw_epipolar_lines(img1, img2, ep11, ep22, keypoints12[0][:20])
    print 'image1 epipolar is ', ep1, ep11
    print 'image2 epipolar is ', ep2, ep22

    errors1 = check_F(F12, matches12[inliers12], keypoints12[0], keypoints12[1])
    print 'inlier mean is ', sum(errors1)/len(errors1)
    print 'inlier min is ', min(errors1)
    print 'inlier max is ', max(errors1)
    print 'inlier var is ', get_var(errors1)

    errors2 = check_F(F12, matches12[np.logical_not(inliers12)], keypoints12[0], keypoints12[1])
    print 'outlier mean is ', sum(errors2)/len(errors2)
    print 'outlier min is ', min(errors2)
    print 'outlier max is ', max(errors2)
    print 'outlier var is ', get_var(errors2)


run()
