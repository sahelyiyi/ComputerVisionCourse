import colorsys
import cv2
import math
import os
import random
import subprocess
import numpy as np

from collections import defaultdict


GRAPH = defaultdict(dict)

BASE_DIR = os.getcwd()

TRAJECTORY_FRAMES_COUNT = 50


def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def extract_video():
    count = 0
    frames_dir = "%s/frames" % BASE_DIR
    make_folder(frames_dir)
    try:
        vidcap = cv2.VideoCapture('video.mp4')
        success, image = vidcap.read()
        success = True
        while success:
            success, image = vidcap.read()
            print('Read a new frame: ', success)
            cv2.imwrite(os.path.join(frames_dir,"frame%d.jpg" % int(count)), image)  # save frame as JPEG file
            count += 1
    except:
        pass
    return count-1


def corner_detection(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    h, w = dst.shape
    sample = dst>0.08*dst.max()
    cnt = 0
    corners = []
    for x in range(0, h):
        for y in range(0, w):
            if sample[x, y]:
                cnt += 1
                corners.append((x, y))
    img[sample]=[0,0,255]
    return corners


def normalize_corners(corners, threshhold=4):
    used = [False] * len(corners)
    unique_corners = defaultdict(list)
    used[0] = True
    for idx, corner1 in enumerate(corners):
        if used[idx]:
            continue
        unique_corners[corner1] = []
        used[idx] = True
        x1, y1 = corner1
        for index, corner2 in enumerate(corners):
            if used[index]:
                continue
            x2, y2 = corner2
            if abs(x1-x2) < threshhold and abs(y1-y2) < threshhold:
                unique_corners[corner1].append(corner2)
                used[index] = True
    return unique_corners.keys()


def get_distance(point1, point2):
    diff_x = abs(point1[0]-point2[0])
    diff_y = abs(point1[1]-point2[1])
    diff = diff_x*diff_x + diff_y*diff_y
    return math.sqrt(diff)


def get_nearest_point(point, corners):
    min_diff = 10000000
    min_point = None
    for corner in corners:
        diff = get_distance(point, corner)
        if diff < min_diff:
            min_diff = diff
            min_point = corner
    return min_point


def match_corners(img1, corners1, img2, corners2, count):
    matches = defaultdict(list)
    for corner1 in corners1:
        x1, y1 = corner1
        for corner2 in corners2:
            x2, y2 = corner2
            if abs(x1-x2) < 1 and abs(y1-y2) < 1:
                continue
            if abs(x1-x2) < max(x1*0.05, 4) and abs(y1-y2) < max(x1*0.05, 4):
                matches[corner1].append(corner2)
        if corner1 in matches and matches[corner1]:
            GRAPH[count][corner1] = get_nearest_point(corner1, matches[corner1])
    return matches


def make_grap(start, end):
    for ii in range(start, end):
        img1 = cv2.imread(os.path.join('%s/frames' % BASE_DIR,'frame%d.jpg' % ii))
        img2 = cv2.imread(os.path.join('%s/frames' % BASE_DIR,'frame%d.jpg' % (ii+1)))
        corners1 = corner_detection(img1)
        corners2 = corner_detection(img2)
        corners1 = normalize_corners(corners1)
        corners2 = normalize_corners(corners2)
        matches = match_corners(img1, corners1, img2, corners2, ii)
        for first in matches:
            if not matches[first]:
                continue
            h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
            color = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
            x1, y1 = first[0], first[1]
            img1 = draw_square(img1, x1, y1, color)
            second = matches[first][0]
            x2, y2 = second[0], second[1]
            img2 = draw_square(img2, x2, y2, color)


def get_max_path(point, h):
    if h in GRAPH.keys():
        if point in GRAPH[h].keys():
            next_point = GRAPH[h][point]
            return 1 + get_max_path(next_point, h+1)
    return 1


def get_intrest_points():
    points_heights = []
    frame_num = 0
    for intrest_point in GRAPH[frame_num]:
        height = get_max_path(intrest_point, 0)
        if height >= len(GRAPH.keys())*0.1:
            points_heights.append((intrest_point, height))
    points_heights = sorted(points_heights, key=lambda x:x[1], reverse=True)

    return [x[0] for x in points_heights]


def normalize_intrest_points(points):
    new_points = []
    for i in range(len(points)):
        check = True
        for point in new_points:
            if get_distance(points[i], point) < 20:
                check = False
        if check:
            new_points.append(points[i])
    return new_points


def get_next_points(points, h):
    if h not in GRAPH.keys():
        return {}
    next_points = {}
    for point in points:
        if point in GRAPH[h].keys():
            next_points[point] = GRAPH[h][point]
    return next_points

def draw_square(img, x, y, color=[255, 0, 0]):
    h, w, _ = img.shape
    for i in range(4):
        for j in range(4):
            if x+i < h and y+j < w:
                img[x+i, y+j] = color
            if x+i < h and y-j >= 0:
                img[x+i, y-j] = color
            if x-i >= 0 and y+j < w:
                img[x-i, y+j] = color
            if x-i >= 0 and y+j >= 0:
                img[x-i, y-j] = color
    return img


def get_trajectory(total_frames):
    img = cv2.imread(os.path.join('%s/frames' % BASE_DIR,'frame0.jpg'))
    cv2.imwrite(os.path.join(BASE_DIR,'trajectory.jpg'),img)
    make_grap(0, total_frames-1)
    points = get_intrest_points()
    points = normalize_intrest_points(points)
    points = points[10:30]
    make_grap(total_frames-1, total_frames*2)
    for ii in range(total_frames*2):
        if ii % 10 == 0:
            print ii
        img = cv2.imread(os.path.join(BASE_DIR,'trajectory.jpg'))
        for point in points:
            x1, y1 = point[0], point[1]
            color = [0,0,255]
            draw_square(img, x1, y1, color)
        cv2.imwrite(os.path.join(BASE_DIR,'trajectory.jpg'),img)
        points = get_next_points(points, ii).values()


total_frames = extract_video()


def a():
    corners_dir = '%s/corners' % BASE_DIR
    frames_dir = '%s/frames' % BASE_DIR
    make_folder(corners_dir)
    for i in range(total_frames):
        img = cv2.imread(os.path.join(frames_dir,'frame%d.jpg' % i))
        corner_detection(img)
        cv2.imwrite(os.path.join(corners_dir, 'corner%d.jpg' % i), img)
    # subprocess.call(['ffmpeg', '-i', '%s/find%d.jpg' % corners_dir, '%s/corners.mp4' % BASE_DIR])


def b():
    get_trajectory(TRAJECTORY_FRAMES_COUNT)


b()