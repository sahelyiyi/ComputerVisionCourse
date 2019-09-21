from copy import deepcopy

import cv2
import numpy as np
from skimage import io
import os
from math import sqrt
from skimage import img_as_float, img_as_ubyte


BASE_DIR = os.getcwd()


def get_sift_homography(img1, img2):
    # X1 = M * X2
    sift = cv2.SIFT()

    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    verify_ratio = 0.8  # Source: stackoverflow
    verified_matches = []
    for m1, m2 in matches:
        if m1.distance < verify_ratio * m2.distance:
            verified_matches.append(m1)

    min_matches = 8
    if len(verified_matches) > min_matches:
        img1_pts = []
        img2_pts = []

        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print 'Error: Not enough matches'
        exit()


def get_warpedimg_size(img1, img2, M):
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
    return x_min, x_max, y_min, y_max


def is_white_float(img, x, y):
    if x < 0 or y < 0:
        return False
    if x >= img.shape[0] or y >= img.shape[1]:
        return False
    r, g, b = img[x, y]
    return (r == 1.0 and g == 1.0 and b == 1.0)


def is_black_float(img, x, y):
    if x < 0 or y < 0:
        return False
    if x >= img.shape[0] or y >= img.shape[1]:
        return False
    r, g, b = img[x, y]
    return (r == 0.0 and g == 0.0 and b == 0.0)


def get_dist(p1, p2):
    diffx = p1[0] - p2[0]
    diffy = p1[1] - p2[1]
    return (sqrt(diffx ** 2 + diffy **2)) + 0.1


def get_nearest_white_up_left(img):
    dp = [[0] * img.shape[1]] * img.shape[0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i==0 or j ==0 or is_black_float(img, i, j):
                dp[i][j] = (i,j)
                continue
            p1, p2 = dp[i-1][j], dp[i][j-1]
            diff1, diff2 = get_dist(p1, (i, j)), get_dist(p2, (i, j))
            dp[i][j] = p1 if diff1 < diff2 else p2

    return dp


def get_nearest_white_up_right(img):
    dp = [[0] * img.shape[1]] * img.shape[0]
    for i in range(img.shape[0] -1 , -1, -1):
        for j in range(img.shape[1]):
            if i== img.shape[0] -1  or j ==0 or is_black_float(img, i, j):
                dp[i][j] = (i,j)
                continue
            p1, p2 = dp[i+1][j], dp[i][j-1]
            diff1, diff2 = get_dist(p1, (i, j)), get_dist(p2, (i, j))
            dp[i][j] = p1 if diff1 < diff2 else p2

    return dp


def get_nearest_white_down_right(img):
    dp = [[0] * img.shape[1]] * img.shape[0]
    for i in range(img.shape[0] -1 , -1, -1):
        for j in range(img.shape[1] -1, -1, -1):
            if i== img.shape[0] -1  or j == img.shape[1] -1 or is_black_float(img, i, j):
                dp[i][j] = (i,j)
                continue
            p1, p2 = dp[i+1][j], dp[i][j+1]
            diff1, diff2 = get_dist(p1, (i, j)), get_dist(p2, (i, j))
            dp[i][j] = p1 if diff1 < diff2 else p2

    return dp


def get_nearest_white_down_left(img):
    dp = [[0] * img.shape[1]] * img.shape[0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1] -1, -1, -1):
            if i== 0  or j == img.shape[1] -1 or is_black_float(img, i, j):
                dp[i][j] = (i,j)
                continue
            p1, p2 = dp[i-1][j], dp[i][j+1]
            diff1, diff2 = get_dist(p1, (i, j)), get_dist(p2, (i, j))
            dp[i][j] = p1 if diff1 < diff2 else p2

    return dp


def get_nearest_white(img):
    dpul = get_nearest_white_up_left(img)
    dpur = get_nearest_white_up_right(img)
    dpdl = get_nearest_white_down_left(img)
    dpdr = get_nearest_white_down_right(img)

    dp = [[0] * img.shape[1]] * img.shape[0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p1, p2, p3, p4 = dpul[i][j], dpur[i][j], dpdl[i][j], dpdr[i][j]
            diff1, diff2 , diff3, diff4 = get_dist(p1, (i, j)), get_dist(p2, (i, j)), get_dist(p3, (i, j)), get_dist(p4, (i, j))
            mn = min(diff1, diff2 , diff3, diff4)
            dp[i][j] = mn
    return dp


def get_pixel_details(images):
    image_dict = [[((0.0, 0.0, 0.0), 0.0)] * images[0].shape[1]] * images[0].shape[0]
    for index, image in enumerate(images):
        nearest_white = get_nearest_white(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not is_black_float(image, i, j):
                    prev_mean, prev_w = image_dict[i][j]
                    prev_b, prev_g, prev_r = prev_mean
                    b, g, r = image[i, j]
                    weight = nearest_white[i][j]
                    new_w = prev_w + weight
                    new_b = (prev_b * prev_w + b * weight) / new_w
                    new_g = (prev_g * prev_w + g * weight) / new_w
                    new_r = (prev_r * prev_w + r * weight) / new_w
                    image_dict[i][j] = ((new_b, new_g, new_r), new_w)
    return image_dict


def float_img(img):
    return img_as_float(img)


def int_img(img):
    return img_as_ubyte(img)


def mean_nomalize_color(result_img, prev_img):
    result_img = float_img(result_img)
    prev_img = float_img(prev_img)

    for i in range(result_img.shape[0]):
       for j in range(result_img.shape[1]):
            r, g, b = result_img[i, j]

            r1, g1, b1 = prev_img[i, j]
            if not is_white_float(prev_img, i, j):
               new_r = (r+r1)/2
               new_g = (g+g1)/2
               new_b = (b+b1)/2
               result_img[i, j] = (new_r, new_g, new_b)

    return int_img(result_img)


def get_stitched_images_mean(img1, imgs, Ms):
    w1, h1 = img1.shape[:2]
    x_min1, x_max1, y_min1, y_max1 = get_warpedimg_size(img1, imgs[0], Ms[0])
    x_min2, x_max2, y_min2, y_max2 = get_warpedimg_size(img1, imgs[-1], Ms[-1])

    x_min = min(x_min1, x_min2)
    x_max = max(x_max1, x_max2)

    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)

    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    result_img = np.zeros([y_max - y_min, x_max - x_min, 3], dtype=np.uint8)
    result_img.fill(255)
    for index, img2 in enumerate(imgs):
        prev_img = deepcopy(result_img)
        result_img = cv2.warpPerspective(img2, transform_array.dot(Ms[index]),
                                         (x_max - x_min, y_max - y_min),
                                         borderMode=cv2.BORDER_TRANSPARENT,
                                         dst=result_img)

        if index > 0:
            result_img = mean_nomalize_color(result_img, prev_img)
    prev_img = deepcopy(result_img)
    result_img[transform_dist[1]:w1 + transform_dist[1], transform_dist[0]:h1 + transform_dist[0]] = img1
    result_img = mean_nomalize_color(result_img, prev_img)
    return result_img


def get_warped_image(img1, imgs, Ms, index):
    x_min1, x_max1, y_min1, y_max1 = get_warpedimg_size(img1, imgs[0], Ms[0])
    x_min2, x_max2, y_min2, y_max2 = get_warpedimg_size(img1, imgs[-1], Ms[-1])

    x_min = min(x_min1, x_min2)
    x_max = max(x_max1, x_max2)

    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)

    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    result_img = np.zeros([y_max - y_min, x_max - x_min, 3], dtype=np.uint8)
    result_img.fill(0)
    return cv2.warpPerspective(imgs[index], transform_array.dot(Ms[index]),
                                         (x_max - x_min, y_max - y_min),
                                         borderMode=cv2.BORDER_TRANSPARENT,
                                         dst=result_img)


def get_stitched_images_blend(pano_images, center_cnt, homographies):
    warped_images = []

    for index, img in enumerate(pano_images):
        img = get_warped_image(pano_images[center_cnt], pano_images, homographies, index)
        warped_images.append(float_img(img))
    pixel_details = get_pixel_details(warped_images)
    print 'pass gettin pixels'
    result_img = np.zeros([warped_images[0].shape[0], warped_images[0].shape[1], 3], dtype=np.uint8)
    result_img.fill(0)
    result_img = float_img(result_img)
    print len(pixel_details), result_img.shape[0]
    print len(pixel_details[0]), result_img.shape[1]
    for i in range(len(pixel_details)):
        for j in range(len(pixel_details[i])):
            color, weight = pixel_details[i][j]
            if weight == 0.0:
                continue
            result_img[i, j] = color
    return int_img(result_img)


def normalize_homographies(homographies, center_cnt):
    # H13 = H12.dot(H23)
    right = [np.identity(3)]
    for i in range(center_cnt, len(homographies)):
        right.append(right[-1].dot(homographies[i]))
    left = [np.identity(3)]
    for i in range(center_cnt - 1, -1, -1):
        left.append(homographies[i].dot(left[-1]))
    left = [np.linalg.inv(x) for x in left[1:]]
    new_homographies = list(reversed(left)) + right
    return new_homographies


def main(type='blend'):
    pano_images = io.ImageCollection(os.path.join(BASE_DIR, 'b*.JPG'))
    pano_images = pano_images[1:]
    center_cnt = 2
    print len(pano_images)
    homographies = []
    for i in range(len(pano_images) - 1):
        img1 = pano_images[i]
        img2 = pano_images[i + 1]
        H = get_sift_homography(img2[:, :, :3], img1[:, :, :3])
        homographies.append(H)
    homographies = normalize_homographies(homographies, center_cnt)
    cv2.imwrite('res10.png', get_stitched_images_mean(pano_images[center_cnt], pano_images, homographies))
    print 'saved res10'
    cv2.imwrite('res11.png', get_stitched_images_blend(pano_images, center_cnt, homographies))
    print 'saved res11'


if __name__ == '__main__':
    main()
