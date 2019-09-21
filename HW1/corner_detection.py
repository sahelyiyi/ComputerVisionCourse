import os
from math import sqrt

from scipy import misc
import numpy as np


BASE_DIR = os.getcwd()
img_name = 'star2.bmp'
new_img_name = 'outfile2.jpg'
WHITE_RANGE = 200


def get_star_center(img):
    h, w, _ = img.shape
    black_cnt = 0
    white_cnt = 0
    x_center = 0
    y_center = 0
    for i in range(0, h):
        for j in range(0, w):
            b, g, r = img[i, j].tolist()
            if b > WHITE_RANGE and g > WHITE_RANGE and r > WHITE_RANGE:
                white_cnt += 1
            else:
                black_cnt += 1
                y_center += j
                x_center += i
    x_center = int(x_center * 1.0 / black_cnt)
    y_center = int(y_center * 1.0 / black_cnt)
    threshhold = int(sqrt(sqrt(black_cnt)))
    return x_center, y_center, threshhold


def normalize_corners(diffs, img, rev, threshhold):
    unique_diffs = []
    unique_diffs.append(diffs[0])
    for i in range(1, len(diffs)):
        _, (x, y) = diffs[i]
        check = -1
        for j in range(len(unique_diffs)):
            diff, (prev_x, prev_y) = unique_diffs[j]
            if abs(x - prev_x) < threshhold and abs(y - prev_y) < threshhold:
                check = j
                continue
        if check == -1:
            unique_diffs.append(diffs[i])
    unique_diffs = sorted(unique_diffs, key=lambda x: x[0], reverse=rev)
    return unique_diffs


def get_corners(img, x_center, y_center, threshhold):
    h, w, _ = img.shape
    distances = []
    for i in range(0, h):
        for j in range(0, w):
            b, g, r = img[i, j].tolist()
            if b > WHITE_RANGE and g > WHITE_RANGE and r > WHITE_RANGE:
                continue
            else:
                diff_x = abs(x_center - i)
                diff_y = abs(y_center - j)
                diff = (diff_x * diff_x) + (diff_y * diff_y)
                distances.append((diff, (i, j)))
    max_distances = sorted(distances, key=lambda x: x[0], reverse=True)
    min_distances = sorted(distances, key=lambda x: x[0], reverse=False)
    maxs = normalize_corners(max_distances, img, True, threshhold)[:5]
    mins = normalize_corners(min_distances, img, False, threshhold)[:5]
    return maxs + mins


def sort_corners(corners):
    sorted_corners = []
    for diff, item in corners:
        angle = np.arctan2(item[0] - x_center, item[1] - y_center)
        sorted_corners.append((angle, diff, item))
    sorted_corners = sorted(sorted_corners, key=lambda x: x[0])
    if sorted_corners[0][1] < sorted_corners[1][1]:
        first = sorted_corners[0]
        sorted_corners = sorted_corners[1:]
        sorted_corners.append(first)
    return sorted_corners


img = misc.imread(os.path.join(BASE_DIR, img_name), flatten=0)
new_img = misc.imread(os.path.join(BASE_DIR, img_name), flatten=0)
x_center, y_center, threshhold = get_star_center(img)
corners = get_corners(img, x_center, y_center, threshhold)
sorted_corners = sort_corners(corners)
for angle, diff, item in sorted_corners:
    print item
h, w, _ = img.shape
color = 200
for angle, diff, item in sorted_corners:
     x = item[0]
     y = item[1]
     for i in range(max(int(h/100.0), 2)):
         for j in range(max(int(h/100.0), 2)):
             if x+i < h and y+j < w:
                 new_img[x+i, y+j] = color
             if x+i < h and y-j >= 0:
                 new_img[x+i, y-j] = color
             if x-i >= 0 and y+j < w:
                 new_img[x-i, y+j] = color
             if x-i >= 0 and y+j >= 0:
                 new_img[x-i, y-j] = color
     color -= 15
misc.imsave(new_img_name, new_img)
