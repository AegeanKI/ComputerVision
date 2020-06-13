import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import vlfeat
from libsvm.svmutil import *


def generate_data(path, normalize=True):
    dir_list = os.listdir(path)
    print('dir_list: ', dir_list)

    data = np.zeros((1, 256))
    label = np.array([])
    for dir in dir_list:
        print('---start reading img data in {}---'.format(dir))
        for img in glob.glob(path + dir + '/*.*'):
            print('img: ', img)
            img_data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            resize_img_data = cv2.resize(img_data, (16, 16)).flatten()

            if normalize:
                # normalize image data
                mean = np.mean(resize_img_data)
                var = np.var(resize_img_data)
                resize_img_data = (resize_img_data - mean) / var

            data = np.vstack((data, resize_img_data))
            label = np.append(label, dir)
            print('data shape: ', data.shape)
            print('label shape: ', label.shape)
    return data[1:], label


def load_data(data_dir, normalize=True):
    training_path = os.path.join(data_dir, 'train/')
    testing_path = os.path.join(data_dir, 'test/')

    if normalize:
        ### normalize ###
        generated_data_path = os.path.join(os.path.dirname(__file__), 'generated_numpy_data/')
        if os.path.isdir(generated_data_path) == False:
            os.mkdir(generated_data_path)

        # Read data from training files
        if os.path.exists(generated_data_path+'train_image.npy') and os.path.exists(generated_data_path+'train_label.npy'):
            train_image = np.load(generated_data_path+'train_image.npy')
            train_label = np.load(generated_data_path+'train_label.npy')
        else:
            train_image, train_label = generate_data(training_path)
            np.save(generated_data_path+'train_image', train_image)
            np.save(generated_data_path+'train_label', train_label)

        # Read data from testing files
        if os.path.exists(generated_data_path+'test_image.npy') and os.path.exists(generated_data_path+'test_label.npy'):
            test_image = np.load(generated_data_path+'test_image.npy')
            test_label = np.load(generated_data_path+'test_label.npy')
        else:
            test_image, test_label = generate_data(testing_path)
            np.save(generated_data_path+'test_image', test_image)
            np.save(generated_data_path+'test_label', test_label)
    else:
        ### un-normalize ###
        generated_data_path = os.path.join(os.path.dirname(__file__), 'generated_unnorm_numpy_data/')
        if os.path.isdir(generated_data_path) == False:
            os.mkdir(generated_data_path)

        # Read data from training files
        if os.path.exists(generated_data_path+'train_image.npy') and os.path.exists(generated_data_path+'train_label.npy'):
            train_image = np.load(generated_data_path+'train_image.npy')
            train_label = np.load(generated_data_path+'train_label.npy')
        else:
            train_image, train_label = generate_data(training_path, normalize=False)
            np.save(generated_data_path+'train_image', train_image)
            np.save(generated_data_path+'train_label', train_label)

        # Read data from testing files
        if os.path.exists(generated_data_path+'test_image.npy') and os.path.exists(generated_data_path+'test_label.npy'):
            test_image = np.load(generated_data_path+'test_image.npy')
            test_label = np.load(generated_data_path+'test_label.npy')
        else:
            test_image, test_label = generate_data(testing_path, normalize=False)
            np.save(generated_data_path+'test_image', test_image)
            np.save(generated_data_path+'test_label', test_label)

    return train_image, train_label, test_image, test_label


class BagOfSift():
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data.astype(np.uint8)
        self.train_label = train_label
        self.test_data = test_data.astype(np.uint8)
        self.test_label = test_label
        # self.train_data = train_data
        # print(np.dtype(self.train_data))

    
    def sift_keypoints(self, data):
        # SIFT extraction
        # print(data[i])
        img_data = np.reshape(data, (16, 16))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = sift.detectAndCompute(img_data, None)
        return descriptors


    def build_vocabulary(self):
        # sift
        sift_keypoints = []
        for i in range(self.train_data.shape[0]): # for each img
            descriptors = self.sift_keypoints(self.train_data[i])
            if descriptors is not None:
                print('{}: descriptors: {}'.format(i, descriptors.shape))
                sift_keypoints.append(descriptors)
        sift_keypoints=np.array(sift_keypoints)
        sift_keypoints=np.concatenate(sift_keypoints, axis=0)

        # kmeans
        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 1000
        _, labels, centers = cv2.kmeans(np.float32(sift_keypoints), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        print('centers: ', centers.shape)
        # print(centers.T)
        print('labels: ', labels.shape)
        # print(labels)
        return labels, centers

    
    def calculate_centroid_histogram(self, voc_centers, img_data):
        # calculate key point
        sift_keypoints = self.sift_keypoints(img_data)
        if sift_keypoints is None:
            return None

        # find voc(aka. center) by nearest neighbor for each keypoint
        nearest_center = np.zeros((sift_keypoints.shape[0], 1))
        for i, point in enumerate(sift_keypoints):
            distance = np.array([])
            for voc in voc_centers:
                distance = np.append(distance, (np.linalg.norm(point - voc)))
            nearest_center[i] = np.argmin(distance)
        # print('sift_keypoint: ', sift_keypoints.shape)
        print('nearest_center: ', nearest_center.shape)

        # build histogram indicating how many times each cluster was used
        voc_labels = np.arange(voc_centers.shape[0]) # voc is 0~(k-1)
        hist, bin_edges = np.histogram(nearest_center, bins=voc_labels)
        print('hist: ', hist.shape)
        return hist

    def main_process(self):
        _, voc = self.build_vocabulary()
        train_hist = []
        available_train_lable = []
        for i in range(self.train_data.shape[0]):
            each_train_hist = self.calculate_centroid_histogram(voc, self.train_data[i])
            if each_train_hist is not None:
                train_hist.append(each_train_hist)
                available_train_lable.append(self.train_label[i])
        train_hist=np.array(train_hist)
        available_train_lable = np.array(available_train_lable)

        test_hist = []
        available_test_lable = []
        for i in range(self.test_data.shape[0]):
            each_test_hist = self.calculate_centroid_histogram(voc, self.test_data[i])
            if each_test_hist is not None:
                test_hist.append(each_test_hist)
                available_test_lable.append(self.test_label[i])
        test_hist=np.array(test_hist)
        available_test_lable = np.array(available_test_lable)
        print('train_hist: ', train_hist.shape)
        print('test_hist: ', test_hist.shape)
        print('\n')
        return train_hist, available_train_lable, test_hist, available_test_lable        


class KNN():
    def __init__(self, k):
        self.k = k

    def gen_distance_matrix(self, train_data, test_data):
        distance_matrix = np.zeros((test_data.shape[0], train_data.shape[0])) # (150, 1500)
        for i in range(test_data.shape[0]):
            for j in range(train_data.shape[0]):
                distance_matrix[i][j] = np.linalg.norm(test_data[i] - train_data[j])
        return distance_matrix

    def vote_for_k_neighbors(self, distance_matrix, train_label):
        result_label = np.array([])
        for i in range(distance_matrix.shape[0]): # for each test image
            # sort by distance and get the first k neighbors' labels
            sort_k_idx = np.argsort(distance_matrix[i])
            k_neighbors_label = list(train_label[sort_k_idx[:self.k]])
            # find most frequent label
            result = max(set(k_neighbors_label), key = k_neighbors_label.count)
            result_label = np.append(result_label, result)
        return result_label

    def calculate_accuracy(self, predict_label, test_label):
        correct = 0
        for i in range(predict_label.shape[0]):
            if predict_label[i] == test_label[i]:
                correct += 1
        return correct / predict_label.shape[0]

    def knn_process(self, train_data, train_label, test_data):
        distance_matrix = self.gen_distance_matrix(train_data, test_data)
        predict_label = self.vote_for_k_neighbors(distance_matrix, train_label)
        return predict_label


class SVM():
    def __init__(self):
        pass

    def train_with_linear_kernel(self, y, x, y_test, x_test):
        param = '-t 0 -h 0'
        model = svm_train(y, x, param)

        print('test:')
        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
        p_label = np.array(p_label)
        return p_label, p_acc
            

if __name__ == "__main__":
    """
    1. Tiny images representation + nearest neighbor classifier

    Uncomment following code to run.
    """
    # train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'))
    # for k in range(1, 22):
    #     knn_Model = KNN(k)
    #     predict = knn_Model.knn_process(train_img, train_label, test_img)
    #     accuracy = knn_Model.calculate_accuracy(predict, test_label)
    #     print('for k={}, accuracy: {}%'.format(k, accuracy*100))


    """
    2. Bag of SIFT representation + nearest neighbor classifier

    Uncomment following code to run.
    """
    # # if we use the normalize data, the sift_keypoints will be None @@
    # train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'), normalize=False)
    
    # bag_sift_Model = BagOfSift(train_img, train_label, test_img, test_label)
    # bag_train_img, bag_train_label, bag_test_img, bag_test_label = bag_sift_Model.main_process()
    # print('bag_train_img: ', bag_train_img.shape)
    # print('bag_train_label: ', bag_train_label.shape)
    # print('bag_test_img: ', bag_test_img.shape)
    # print('bag_test_label: ', bag_test_label.shape)

    # for k in range(1, 22):
    #     knn_Model = KNN(k)
    #     predict = knn_Model.knn_process(bag_train_img, bag_train_label, bag_test_img)
    #     accuracy = knn_Model.calculate_accuracy(predict, bag_test_label)
    #     print('for k={}, accuracy: {}%'.format(k, accuracy*100))


    """
    3. Bag of SIFT representation + linear SVM classifier

    Uncomment following code to run.
    """
    label_dict = {
        'Bedroom': 0,
        'Coast': 1, 
        'Forest': 2, 
        'Highway': 3, 
        'Industrial': 4, 
        'InsideCity': 5, 
        'Kitchen': 6, 
        'LivingRoom': 7, 
        'Mountain': 8, 
        'Office': 9, 
        'OpenCountry': 10, 
        'Store': 11, 
        'Street': 12, 
        'Suburb': 13, 
        'TallBuilding': 14
    }
    train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'), normalize=False)

    # turn string label into int label
    train_label_int = []
    for str_label in train_label:
        train_label_int.append(int(label_dict[str_label]))
    
    test_label_int = []
    for str_label in test_label:
        test_label_int.append(int(label_dict[str_label]))

    train_label_int = np.array(train_label_int)
    test_label_int = np.array(test_label_int)
    

    bag_sift_Model = BagOfSift(train_img, train_label_int, test_img, train_label_int)
    bag_train_img, bag_train_label, bag_test_img, bag_test_label = bag_sift_Model.main_process()

    svm_Model = SVM()
    predict, accuracy = svm_Model.train_with_linear_kernel(bag_train_label, bag_train_img, bag_test_label, bag_test_img)
