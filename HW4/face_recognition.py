import os
import cv2
import json
import math
import numpy as np

from collections import defaultdict

import operator

BASE_DIR = os.getcwd() + '/pics02'


def read_images():
    train = defaultdict(list)
    valid = defaultdict(list)
    test = defaultdict(list)
    for image_file_name in os.listdir(BASE_DIR):
        person_num, image_num = image_file_name.split('_')
        person_num = person_num[6:]
        img = cv2.imread(os.path.join(BASE_DIR, image_file_name), 0)
        train[person_num].append(np.array(img, dtype='float64').flatten())
    validation_percentage = 0.15
    test_percentage = 0.15
    for person in train:
        person_images_cnt = len(train[person])
        validation_cnt = int(validation_percentage * person_images_cnt)
        valid[person] = train[person][:validation_cnt]
        test[person] = train[person][validation_cnt:validation_cnt+int(test_percentage * person_images_cnt)]
        train[person] = train[person][int((validation_percentage + test_percentage) * person_images_cnt):]

    return train, valid, test


def get_cov_matrix(images):
    data_matrix = np.empty(shape=(images[0][0].shape[0], len(images)), dtype='float64')
    for index, (image, person) in enumerate(images):
        data_matrix[:, index] = image[:]
    mean_img = np.sum(data_matrix, axis=1) / len(images)
    for j in xrange(0, len(images)):
        data_matrix[:, j] -= mean_img[:]
    cov_mat = np.matrix(data_matrix) * np.matrix(data_matrix.transpose())
    cov_mat /= data_matrix.shape[0]
    return cov_mat, mean_img


def pca(X):
    eig_vals, eig_vecs = np.linalg.eigh(X)
    for index, ev in enumerate(eig_vecs):
        normalize_ev = np.divide(ev, np.linalg.norm(ev))
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(normalize_ev))
        eig_vecs[index] = normalize_ev
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    for idx, (eig_val, eig_vec) in enumerate(eig_pairs):
        eig_vecs[:, idx] = eig_vec
    return eig_vecs


def get_dis(img1, img2):
    dis = 0.0
    for i in range(len(img1)):
        first = img1[i]
        second = 0
        if i < len(img2):
            second = img2[i]
        dis += ((first - second) ** 2)
    return math.sqrt(dis) / len(img1)


def calculate_score(selected_vecs, mean, train_images, images):
    score = 0.0
    train_ws = []
    for image, person in train_images:
        img = image - mean
        w = selected_vecs.transpose().dot(img)
        train_ws.append((w, person))
    for image, person in images:
        img = image - mean
        w = selected_vecs.transpose().dot(img)
        img_dises = []
        for tw, tperson in train_ws:
            img_dises.append((get_dis(tw, w), tperson))
        img_dises = sorted(img_dises, key=lambda tup: tup[0])
        if img_dises[0][1] == person:
            score += 1
    return score / len(images)


def train(eig_vecs, mean, train_images, validation_images):
    scores = {}
    for i in range(1, 10):
        d = 2 ** i
        selected_vecs = eig_vecs[:, :d]
        scores[d] = calculate_score(selected_vecs, mean, train_images, validation_images)
    return sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def get_all_class_images(images):
    all_images = []
    for person in images:
        for person_image in images[person]:
            all_images.append((person_image, person))
    return all_images


def main():
    train_images, validation_images, test_images = read_images()
    all_train_images = get_all_class_images(train_images)
    all_validation_images = get_all_class_images(validation_images)
    all_test_images = get_all_class_images(test_images)

    cov_mat, mean_img = get_cov_matrix(all_train_images)
    cov_mat = np.array(cov_mat)
    mean_img = np.array(mean_img)
    show_mean = mean_img.reshape((50, 50))
    cv2.imwrite('mean_face.jpg', show_mean)

    eig_vecs = pca(cov_mat)
    selected_eig = eig_vecs[:5, :]
    with open('eig_vectors', 'w') as f:
        f.write(json.dumps(selected_eig.tolist()))
    cnt = 0
    for eig in selected_eig:
        cnt += 1
        eig = 255*eig
        cv2.imwrite('eig_faces%d.jpg' % cnt, eig.reshape((50, 50)))
    eig_vecs = np.array(eig_vecs)

    selected_d = train(eig_vecs, mean_img, all_train_images, all_validation_images)
    print 'selected d is ', selected_d
    selected_vecs = eig_vecs[:, :selected_d]
    print 'score is ', calculate_score(selected_vecs, mean_img, all_train_images, all_test_images)


if __name__ == '__main__':
    main()
