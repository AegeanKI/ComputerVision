import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import glob


def generate_data(path):
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
            data = np.vstack((data, resize_img_data))
            label = np.append(label, dir)
            print('data shape: ', data.shape)
            print('label shape: ', label.shape)
    return data[1:], label


def load_data(data_dir):
    training_path = os.path.join(data_dir, 'train/')
    testing_path = os.path.join(data_dir, 'test/')

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

    return train_image, train_label, test_image, test_label


class knn():
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

    def calculate_accuracy(self, result_label, test_label):
        correct = 0
        for i in range(result_label.shape[0]):
            if result_label[i] == test_label[i]:
                correct += 1
        return correct / result_label.shape[0]

    def knn_process(self, train_data, train_label, test_data, test_label):
        distance_matrix = self.gen_distance_matrix(train_data, test_data)
        result_label = self.vote_for_k_neighbors(distance_matrix, train_label)
        return result_label
            

if __name__ == "__main__":
    train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'))
    # print('train_image: ', train_img.shape)
    # print('train_label: ', train_label.shape)
    # print('test_image: ', test_img.shape)
    # print('test_label: ', test_label.shape)

    KNN_Model = knn(8)
    result = KNN_Model.knn_process(train_img, train_label, test_img, test_label)
    accuracy = KNN_Model.calculate_accuracy(result, test_label)
    print('accuracy: {}%'.format(accuracy*100))