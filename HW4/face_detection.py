from numpy import *
import numpy as np
import json
import cv2
import os


BASE_DIR = os.getcwd()
IMG_SIZE = 24

USED_FITURES = []
# HAARS = []


def hash_haar_square():
    haars = []
    for y_scale in range(1, IMG_SIZE):
        for x_scale in range(1, IMG_SIZE):
            for i in range(IMG_SIZE):
                if i + 2 * y_scale >= IMG_SIZE:
                    break
                for j in range(IMG_SIZE):
                    if j + 2 * x_scale >= IMG_SIZE:
                        break
                    haars.append(((i, j), (y_scale, x_scale), 'haar_square', ''))
    return haars


def hash_haar_triple():
    haars = []
    for y_scale in range(1, IMG_SIZE):
        for x_scale in range(1, IMG_SIZE):
            for i in range(IMG_SIZE):
                if i + y_scale >= IMG_SIZE:
                    break
                for j in range(IMG_SIZE):
                    right_flag = False
                    down_flag = False
                    if i + y_scale < IMG_SIZE and j + 3 * x_scale < IMG_SIZE:
                        right_flag = True
                    if i + 3 * y_scale < IMG_SIZE and j + x_scale < IMG_SIZE:
                        down_flag = True
                    if not (right_flag or down_flag):
                        break
                    if right_flag:
                        haars.append(((i, j), (y_scale, x_scale), 'haar_triple', 'right_flag'))
                    if down_flag:
                        haars.append(((i, j), (y_scale, x_scale), 'haar_triple', 'down_flag'))
    return haars


def hash_haar_domino():
    haars = []
    for y_scale in range(1, IMG_SIZE):
        for x_scale in range(1, IMG_SIZE):
            for i in range(IMG_SIZE):
                if i + y_scale >= IMG_SIZE:
                    break
                for j in range(IMG_SIZE):
                    right_flag = False
                    down_flag = False
                    if i + y_scale < IMG_SIZE and j + 2 * x_scale < IMG_SIZE:
                        right_flag = True
                    if i + 2 * y_scale < IMG_SIZE and j + x_scale < IMG_SIZE:
                        down_flag = True
                    if not (right_flag or down_flag):
                        break
                    if right_flag:
                        haars.append(((i, j), (y_scale, x_scale), 'haar_domino', 'right_flag'))
                    if down_flag:
                        haars.append(((i, j), (y_scale, x_scale), 'haar_domino', 'down_flag'))
    return haars


def hash_haars():
    haars = hash_haar_square()
    haars += hash_haar_triple()
    haars += hash_haar_domino()
    return haars


def read_images(directory):
    integral_images = []
    imgs_dir = os.path.join(BASE_DIR, directory)
    for image_file_name in os.listdir(imgs_dir):
        img = cv2.imread(os.path.join(imgs_dir, image_file_name))
        try:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        except:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        integral_img = cv2.integral(img)
        integral_images.append(integral_img)
    return integral_images


def get_window_score(integral_image, top_left, bottom_right):
    top_right = (top_left[0], bottom_right[1])
    bottom_left = (bottom_right[0], top_left[1])
    top_left = (top_left[0] - 1, top_left[1] - 1)
    top_right = (top_right[0] - 1, top_right[1])
    bottom_left = (bottom_left[0], bottom_left[1] - 1)
    return integral_image[bottom_right] - integral_image[bottom_left] - integral_image[top_right] + integral_image[
        top_left]


def convert_original_coordinate_to_integral(pixel):
    return (pixel[0] + 1, pixel[1] + 1)


def get_bottom_right(top_left, x_scale, y_scale):
    return (top_left[0] + y_scale - 1, top_left[1] + x_scale - 1)


def get_right_block(pixel, x_scale, y_scale):
    top_left = (pixel[0], pixel[1] + x_scale)
    bottom_right = (pixel[0] + y_scale - 1, pixel[1] + 2 * x_scale - 1)
    return top_left, bottom_right


def get_down_block(pixel, x_scale, y_scale):
    top_left = (pixel[0] + y_scale, pixel[1])
    bottom_right = (pixel[0] + 2 * y_scale - 1, pixel[1] + x_scale - 1)
    return top_left, bottom_right


def get_right_down_block(pixel, x_scale, y_scale):
    top_left = (pixel[0] + y_scale, pixel[1] + x_scale)
    bottom_right = (pixel[0] + 2 * y_scale - 1, pixel[1] + 2 * x_scale - 1)
    return top_left, bottom_right


def haar_square(integral_img, i, j, y_scale, x_scale):
    top_left1 = convert_original_coordinate_to_integral((i, j))
    bottom_right1 = get_bottom_right(top_left1, x_scale, y_scale)
    center = get_window_score(integral_img, top_left1, bottom_right1)

    top_left2, bottom_right2 = get_right_block(top_left1, x_scale, y_scale)
    right = get_window_score(integral_img, top_left2, bottom_right2)

    top_left3, bottom_right3 = get_down_block(top_left1, x_scale, y_scale)
    down = get_window_score(integral_img, top_left3, bottom_right3)

    top_left4, bottom_right4 = get_right_down_block(top_left1, x_scale, y_scale)
    down_right = get_window_score(integral_img, top_left4, bottom_right4)
    # [[1, -1]
    # [-1, 1]]

    return int(center - right - down + down_right)


def haar_triple(image, i, j, y_scale, x_scale, flag):
    top_left1 = convert_original_coordinate_to_integral((i, j))
    bottom_right1 = get_bottom_right(top_left1, x_scale, y_scale)
    center = get_window_score(image, top_left1, bottom_right1)
    if flag == 'right_flag':
        top_left21, bottom_right21 = get_right_block(top_left1, x_scale, y_scale)
        right1 = get_window_score(image, top_left21, bottom_right21)

        top_left22, bottom_right22 = get_right_block(top_left1, x_scale, y_scale)
        right2 = get_window_score(image, top_left22, bottom_right22)
        # [[1, -1, 1]]
        return int(center - right1 + right2)
    if flag == 'down_flag':
        top_left31, bottom_right31 = get_down_block(top_left1, x_scale, y_scale)
        down1 = get_window_score(image, top_left31, bottom_right31)

        top_left32, bottom_right32 = get_down_block(top_left1, x_scale, y_scale)
        down2 = get_window_score(image, top_left32, bottom_right32)
        # [[1],
        #  [-1],
        #  [1]]
        return int(center - down1 + down2)


def haar_domino(image, i, j, y_scale, x_scale, flag):
    top_left1 = convert_original_coordinate_to_integral((i, j))
    bottom_right1 = get_bottom_right(top_left1, x_scale, y_scale)
    center = get_window_score(image, top_left1, bottom_right1)
    if flag == 'right_flag':
        top_left2, bottom_right2 = get_right_block(top_left1, x_scale, y_scale)
        right = get_window_score(image, top_left2, bottom_right2)
        # [[1, -1]]
        return int(center - right)
    if flag == 'down_flag':
        top_left3, bottom_right3 = get_down_block(top_left1, x_scale, y_scale)
        down = get_window_score(image, top_left3, bottom_right3)
        # [[-1],
        #  [1]]
        return int(-center + down)


def train_classify(data, num, threshold, comp_factor):
    res = ones((shape(data)[0], 1))
    if comp_factor == 'lt':
        res[data[:, num] <= threshold] = -1.0
    else:
        res[data[:, num] > threshold] = -1.0
    return res


def get_best_classifier(X, y, w):
    X_matrix = mat(X)
    y_matrix = mat(y).T
    m, n = shape(X_matrix)
    step_num = 10.0
    best = {}
    best_class = mat(zeros((5, 1)))
    min_error = inf
    for i in range(n):
        if i in USED_FITURES:
            continue
        Xmin = X_matrix[:, i].min()
        Xmax = X_matrix[:, i].max()
        step_size = (Xmax - Xmin) / step_num
        for j in range(-1, int(step_num) + 1):
            for comp_factor in ['lt', 'gt']:
                threshold = Xmin + float(j) * step_size
                predict = train_classify(X_matrix, i, threshold, comp_factor)
                error = mat(ones((m, 1)))
                error[predict == y_matrix] = 0
                weighted_err = w.T * error
                if weighted_err < min_error:
                    min_error = weighted_err
                    best_class = predict.copy()
                    best['num'] = i
                    best['threshold'] = threshold
                    best['comp_factor'] = comp_factor
    return best, min_error, best_class


def train(X, y, iter_num):
    weak_classifiers = []
    size = shape(X)[0]
    w = mat(ones((size, 1)) / size)
    imgs_scores = mat(zeros((size, 1)))
    for i in range(iter_num):
        best, error, img_scores = get_best_classifier(X, y, w)
        alpha = float(0.5 * log((1.0 - error) / (error + 1e-15)))
        best['alpha'] = alpha
        USED_FITURES.append(best['num'])
        weak_classifiers.append(best)
        new_w = multiply((-1 * alpha * mat(y)).T, img_scores)
        w = multiply(w, exp(new_w))
        w = w / w.sum()
        imgs_scores += img_scores * alpha
        print ('train ', i)
    return weak_classifiers


def _cal_haar(integral_img, haar):
    (x, y), (y_scale, x_scale), haar_type, flag = haar
    if haar_type == 'haar_square':
        return haar_square(integral_img, x, y, y_scale, x_scale)
    if haar_type == 'haar_triple':
        return haar_triple(integral_img, x, y, y_scale, x_scale, flag)
    if haar_type == 'haar_domino':
        return haar_domino(integral_img, x, y, y_scale, x_scale, flag)


def calculate_haar(faces, non_faces, haars):
    print (len(faces), len(non_faces))
    face_haars = []
    classes = []
    for image in faces:
        haar_scores = []
        for haar in haars:
            haar_scores.append(_cal_haar(image, haar))
        face_haars.append(haar_scores)
        classes.append(1)
        if len(classes) % 10 == 0:
            print len(classes)

    non_face_haars = []
    for image in non_faces:
        haar_scores = []
        for haar in haars:
            haar_scores.append(_cal_haar(image, haar))
        non_face_haars.append(haar_scores)
        classes.append(0)
        if len(classes) % 10 == 0:
            print len(classes)

    all_haars = np.array(face_haars + non_face_haars)
    classes = np.array(classes)
    return all_haars, classes


def classify(data, threshold, comp_factor='lt'):
    if comp_factor == 'lt':
        if data <= threshold:
            return -1.0
        else:
            return 1.0
    else:
        if data > threshold:
            return -1.0
        else:
            return 1.0


def calculate_adaboost(img_selected_haars, classifiers):
    alphas = 0.0
    img_scores = 0.0
    for i in range(len(classifiers)):
        img_score = classify(img_selected_haars[i], classifiers[i]['threshold'], classifiers[i]['comp_factor'])
        img_scores += classifiers[i]['alpha'] * img_score
        alphas += classifiers[i]['alpha']
    return img_scores < alphas * 0.5


def extract_faces(img, casecades, haars):
    selected_haars = []
    for casecade in casecades:
        this_haars = []
        for item in casecade:
            this_haars.append(item['num'])
        selected_haars.append(this_haars)
    new_img = img
    for i in range(10):
        for j in range(new_img.shape[0] - IMG_SIZE):
            for k in range(new_img.shape[1] - IMG_SIZE):
                croped_img = new_img[j:j + IMG_SIZE, k:k + IMG_SIZE]
                integral_img = cv2.integral(croped_img)
                face_flag = True
                for index, casecade in enumerate(casecades):
                    haar_scores = []
                    for selected_haar in selected_haars[index]:
                        haar_scores.append(_cal_haar(integral_img, haars[selected_haar]))
                    if calculate_adaboost(haar_scores, casecade):
                        face_flag = False
                        break
                if face_flag:
                    face_num = len(os.listdir(os.path.join(BASE_DIR, 'faces'))) + 1
                    cv2.imwrite(os.path.join(BASE_DIR, 'faces', '%d.png' % face_num), croped_img)

        new_img = cv2.resize(new_img, (int(new_img.shape[0] * 0.7), int(new_img.shape[1] * 0.7)))


def main():
    if not os.path.exists(os.path.join(BASE_DIR, 'faces')):
        os.mkdir(os.path.join(BASE_DIR, 'faces'))
    face_images = read_images('pics01')
    face_images = face_images[:15]
    # test_faces = face_images[1500:1550]
    non_face_images = read_images('non_face')
    non_face_images = non_face_images[:10]
    # test_nonfaces = non_face_images[1000:1050]
    print ('pass loading images')
    haars = hash_haars()
    print ('pass hash haars')
    X, y = calculate_haar(face_images, non_face_images, haars)
    print ('pass calculate haars')
    cascades = []
    # for i in range(10):
    for i in range(1):
        casecade = train(X, y, 1)
        # casecade = train(X, y, 4)
        cascades.append(casecade)
        print('cascade ', i)
    test_img = cv2.imread(os.path.join(BASE_DIR, 'test01.jpg'), 0)
    extract_faces(test_img, cascades, haars)
    print ('finished')


if __name__ == '__main__':
    main()
