import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from cyvlfeat.kmeans import kmeans
from cyvlfeat.sift import dsift
from libsvm.svmutil import *
import threading
import concurrent.futures
import time
import math


def generate_data(path, resize=False, normalize=True):
    dir_list = os.listdir(path)
    print('dir_list: ', dir_list)
    
    data = []
    label = np.array([])
    for dir in dir_list:
        print(f'---start reading img data in {dir}---')
        i = 0
        for img in glob.glob(path + dir + '/*.*'):
            print('img: ', img,end='\r')
            img_data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            if resize:
                img_data = cv2.resize(img_data, (resize, resize))

            if normalize:
                mean = np.mean(img_data)
                var = np.var(img_data)
                img_data = (img_data - mean) / var

            data.append(np.array(img_data))
            label = np.append(label, dir)
            i += 1
    return data, label


def load_data(data_dir, resize=False, normalize=True):
    training_path = os.path.join(data_dir, 'train/')
    testing_path = os.path.join(data_dir, 'test/')

    generate_data_path = None
    test_image, test_label = generate_data(testing_path, resize, normalize)
    train_image, train_label = generate_data(training_path, resize, normalize)

    return train_image, train_label, test_image, test_label


class BagOfSift():
    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        # self.train_data = train_data
        # print(np.dtype(self.train_data))

    
    def sift_keypoints(self, data):
        _, descriptors = dsift(data, fast=True, step=10)
        return descriptors


    def build_vocabulary(self, limit=True):
        # sift
        sift_keypoints = []
        n_each = math.ceil(10000 / len(self.train_data))
        for i in range(len(self.train_data)): # for each img
            descriptors = self.sift_keypoints(self.train_data[i])
            if descriptors is not None:
                print(f'descriptors of img {i}: {descriptors.shape}', end='\r')
                if limit and descriptors.shape[0] > n_each:
                    idx = np.random.choice(descriptors.shape[0], n_each, replace=False)
                    sift_keypoints.append(descriptors[idx])
                else:
                    sift_keypoints.append(descriptors)
        sift_keypoints=np.array(sift_keypoints)
        sift_keypoints=np.concatenate(sift_keypoints, axis=0)

        print(f'\ndescriptors num: {sift_keypoints.shape[0]}')
        # define K centers, which is K vocabulary
        k = 20
        centers = kmeans(data=np.float32(sift_keypoints), num_centers=k, initialization="PLUSPLUS")
        print(f'centers: {centers.shape}')
        return centers

    
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
                voc_distance = np.linalg.norm(point - voc)
                distance = np.append(distance, voc_distance)
                if voc_distance == 0:
                    break
            # record the least distance center
            nearest_center[i] = np.argmin(distance)

        # build histogram indicating how many times each cluster was used
        voc_labels = np.arange(voc_centers.shape[0]+1) # voc is 0~(k-1)
        hist, bin_edges = np.histogram(nearest_center, bins=voc_labels)
        return hist

    def calculate_centroid_histogram_thread(self,voc_centers, img_datas, start, end):
        hist_list = []
        for idx in range(start,end):
            # calculate key point
            sift_keypoints = self.sift_keypoints(img_datas[idx])
            if sift_keypoints is None:
                return None

            # find voc(aka. center) by nearest neighbor for each keypoint
            nearest_center = np.zeros((sift_keypoints.shape[0], 1))
            for i, point in enumerate(sift_keypoints):
                distance = np.array([])
                for voc in voc_centers:
                    voc_distance = np.linalg.norm(point - voc)
                    distance = np.append(distance, voc_distance)
                    if voc_distance == 0:
                        break
                # record the least distance center
                nearest_center[i] = np.argmin(distance)
            print(f'idx >> {idx} << nearest_center shape: {nearest_center.shape}', end='\r')

            # build histogram indicating how many times each cluster was used
            voc_labels = np.arange(voc_centers.shape[0]+1) # voc is 0~(k-1)
            hist, bin_edges = np.histogram(nearest_center, bins=voc_labels)
            hist_list.append(hist)
        return hist_list


    def main_process(self):
        voc = self.build_vocabulary()
        train_hist = []
        available_train_lable = []
        # Try to use multi process to run faster
        futures = []
        max = 30
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(max):
                start = 0 + i*len(self.train_data)//max
                end = (i+1)*len(self.train_data)//max
                print(f'process start, start: {start}, end: {end}')
                futures.append(executor.submit(self.calculate_centroid_histogram_thread,voc,self.train_data,start,end))
        
            # wait for process
            for future in futures:
                train_hist+=future.result()
        
        print(f'\ntrain hist shape: {len(train_hist)}, {len(train_hist[0])}')
        available_train_lable = self.train_label

        test_hist = []
        available_test_lable = []
        
        # Try to use multi process to run faster
        futures = []
        max = 10
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(max):
                start = 0 + i*len(self.test_data)//max
                end = (i+1)*len(self.test_data)//max
                print(f'process start, start: {start}, end: {end}')
                futures.append(executor.submit(self.calculate_centroid_histogram_thread,voc,self.test_data,start,end))
        
            # wait for threads
            for future in futures:
                test_hist+=future.result()

        print("\nBag of SIFT,K = ",len(train_hist))
        available_test_lable = self.test_label
        return train_hist, available_train_lable, test_hist, available_test_lable        



class KNN():
    def __init__(self, k):
        self.k = k

    def chi_sqr(self,h1,h2):
        d = 0.5 * np.sum([((a - b)**2 ) for (a,b) in zip(h1,h2)])
        return d

    def gen_distance_matrix(self, train_data, test_data, Hist):
        distance_matrix = np.zeros((len(test_data), len(train_data))) # (150, 1500)
        for i in range(len(test_data)):
            for j in range(len(train_data)):
                if Hist:
                    # cv2 compare hist only eats float32
                    distance_matrix[i][j] = cv2.compareHist(test_data[i].flatten().astype('float32'), train_data[j].flatten().astype('float32'),cv2.HISTCMP_CHISQR)
                    # distance_matrix[i][j] = self.chi_sqr(test_data[i],train_data[j])
                else: 
                    distance_matrix[i][j] = np.linalg.norm(test_data[i] - train_data[j] )
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

    def knn_process(self, train_data, train_label, test_data, Hist=False):
        distance_matrix = self.gen_distance_matrix(train_data, test_data, Hist)
        predict_label = self.vote_for_k_neighbors(distance_matrix, train_label)
        return predict_label


class SVM():
    def __init__(self):
        pass

    def train_with_linear_kernel(self, y, x, y_test, x_test):
        param = '-t 0 -h 0'
        # prob = svm_problem(y,x)
        model = svm_train(y,x, param)

        print('test:')
        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
        p_label = np.array(p_label)
        return p_label, p_acc
            

if __name__ == "__main__":
    """
    1. Tiny images representation + nearest neighbor classifier

    Uncomment following code to run.
    """
    # train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'), resize=16, normalize=True)
    # for k in range(1, 22):
    #     knn_Model = KNN(k)
    #     predict = knn_Model.knn_process(train_img, train_label, test_img)
    #     accuracy = knn_Model.calculate_accuracy(predict, test_label)
    #     print('for k={}, accuracy: {}%'.format(k, accuracy*100))


    """
    2. Bag of SIFT representation + nearest neighbor classifier

    Uncomment following code to run.
    """

    # if we use the normalize data, the sift_keypoints will be None @@
    start_time = time.time()
    train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'), resize=False, normalize=False)
    bag_sift_Model = BagOfSift(train_img, train_label, test_img, test_label)
    bag_train_hist, bag_train_label, bag_test_hist, bag_test_label = bag_sift_Model.main_process()
    print("\ntime eslaped: {:.2f}s = {:.2f} mins".format(time.time() - start_time,(time.time()-start_time)/60))

    # for k in range(1, 22):
    #     knn_Model = KNN(k)
    #     predict = knn_Model.knn_process(bag_train_hist, bag_train_label, bag_test_hist,Hist=True)
    #     accuracy = knn_Model.calculate_accuracy(predict, bag_test_label)
    #     print('for k={}, accuracy: {:.2f}%'.format(k, accuracy*100))


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
    train_img, train_label, test_img, test_label = load_data(os.path.join(os.path.dirname(__file__), 'hw5_data/'), resize=False, normalize=False)

    # turn string label into int label
    train_label_int = []
    for str_label in train_label:
        train_label_int.append(int(label_dict[str_label]))
    
    test_label_int = []
    for str_label in test_label:
        test_label_int.append(int(label_dict[str_label]))

    train_label_int = list(train_label_int)
    test_label_int = list(test_label_int)
    
    start_time = time.time()

    bag_sift_Model = BagOfSift(train_img, train_label_int, test_img, train_label_int)
    bag_train_hist, bag_train_label, bag_test_hist, bag_test_label = bag_sift_Model.main_process()
    bag_test_hist = np.array(bag_test_hist).tolist()
    bag_train_hist = np.array(bag_train_hist).tolist()
    print('bag_train_label: ',type(bag_train_label), len(bag_train_label))
    print('bag_train_hist: ',type(bag_train_hist[0]), len(bag_train_hist),len(bag_train_hist[0]))
    print('bag_test_label: ',type(bag_test_label), len(bag_test_label))
    print('bag_test_hist: ', type(bag_test_hist[0][1]),len(bag_test_hist),len(bag_test_hist[0]))
    print("time eslaped:{:.2f}s , {:.2f}mins".format(time.time() - start_time,(time.time()-start_time)/60))
    svm_Model = SVM()
    predict, accuracy = svm_Model.train_with_linear_kernel(bag_train_label, bag_train_hist, bag_test_label, bag_test_hist)
