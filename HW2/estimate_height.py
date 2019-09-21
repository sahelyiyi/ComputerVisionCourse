from skimage import feature, color, transform, io
import numpy as np
import sys
import cv2
from math import sqrt


def get_edges(image, sigma=3):
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=3,
                                               line_gap=2)

    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def get_edge_lines(edges):
    locations, directions, _ = edges
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def get_inliers(edges, vp, threshold_inlier=5):
    vp = vp[:2] / vp[2]

    locations, directions, strengths = edges

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths


def get_vanishing_point(edges, num_ransac_iter=2000, threshold_inlier=5):
    locations, directions, strengths = edges
    lines = get_edge_lines(edges)

    strenghths_cnt = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:strenghths_cnt // 5]
    second_index_space = arg_sort[:strenghths_cnt // 2]

    best_vp = None
    best_inliers = np.zeros(strenghths_cnt)

    for ransac_iter in range(num_ransac_iter):
        first_edge = np.random.choice(first_index_space)
        second_edge = np.random.choice(second_index_space)

        l1 = lines[first_edge]
        l2 = lines[second_edge]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            continue

        current_inliers = get_inliers(
            edges, current_model, threshold_inlier)

        if current_inliers.sum() > best_inliers.sum():
            best_vp = current_model
            best_inliers = current_inliers

    return best_vp


def make_model_better(model, edges, threshold_reestimate=5):
    locations, directions, strengths = edges

    inliers = get_inliers(edges, model, threshold_reestimate) > 0
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = get_edge_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    est_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((est_model, [1.]))


def remove_inliers(model, edges, threshold_inlier=10):
    inliers = get_inliers(edges, model, threshold_inlier) > 0
    locations, directions, strengths = edges
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edges = (locations, directions, strengths)
    return edges


def get_dis(x1, y1, x2, y2):
    diff_X = abs(x2 - x1)
    diff_y = abs(y2 - y1)
    return sqrt(diff_X * diff_X + diff_y * diff_y)


def get_vanishing_points(img_name):
    img = io.imread(img_name)
    first_edges = get_edges(img)
    vp1 = get_vanishing_point(first_edges, num_ransac_iter=2000,
                              threshold_inlier=0.5)
    vp1 = make_model_better(vp1, first_edges, threshold_reestimate=0.5)

    second_edges = remove_inliers(vp1, first_edges, 5)
    vp2 = get_vanishing_point(second_edges, num_ransac_iter=2000,
                              threshold_inlier=0.1)
    vp2 = make_model_better(vp2, second_edges, threshold_reestimate=0.1)

    third_edges = remove_inliers(vp2, second_edges, 5)
    vp3 = get_vanishing_point(third_edges, num_ransac_iter=2000,
                              threshold_inlier=0.1)
    vp3 = make_model_better(vp3, third_edges, threshold_reestimate=0.1)
    return vp1, vp2, vp3


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


def find_fixed_points(image):
    corners = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            if r >= 250 and b <= 10 and g <= 10:
                check = True
                for corner in corners:
                    if get_dis(i, j, corner[0], corner[1]) < 20:
                        check = False
                if check:
                    corners.append((i, j))
    return corners


def estimate(image, vp1, vp2, vp3):
    corners = find_fixed_points(image)
    wall_top, woman_top, wall_bottom, woman_bottom = corners
    vz_x, vz_y = line_intersection((wall_bottom, wall_top), (vp1, vp2))
    horizon_x, horizon_y = line_intersection((wall_bottom, woman_bottom), (vp1, vp2))
    woman_image_x, woman_image_y = line_intersection(((horizon_x, horizon_y), woman_top), (wall_bottom, wall_top))
    real_woman_height = 175.0
    woman_height = get_dis(woman_image_x, woman_image_y, *wall_bottom)
    wall_height = get_dis(*(wall_bottom + wall_top))
    real_wall_height = real_woman_height * 1.0 * (wall_height * get_dis(vz_x, vz_y, woman_image_x, woman_image_y)) / (woman_height * get_dis(vz_x, vz_y, *wall_top))
    return real_wall_height


if __name__ == '__main__':
    img_name = '311.JPG'
    if len(sys.argv)> 1 and sys.argv[1] == 'manual':
        img_name = '411.JPG'
    image = cv2.imread(img_name)
    # vp1 = [-7.01662549e+03, 7.43210886e+02, 1.00000000e+00]
    # vp2 = [727.63986943, 498.61887181, 1.0]
    vp1, vp2, vp3 = get_vanishing_points(img_name)
    H1 = estimate(image, vp1, vp2, vp3)
    H2 = estimate(image, vp1, vp3, vp2)
    H3 = estimate(image, vp2, vp3, vp1)
    print max([H1, H2, H3])


