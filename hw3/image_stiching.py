import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = 'data/'
IMAGES = [["1.jpg", "2.jpg"], ["hill1.JPG", "hill2.JPG"], ["S1.jpg", "S2.jpg"]]
GOOD_MATCH_K = 2
GOOD_DISTANCE_RATIO = 0.3

def find_img_keypoint(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des

def find_good_matches(des_0, des_1):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_0, des_1, k=GOOD_MATCH_K)
    good = []
    for m,n in matches:
        if m.distance < GOOD_DISTANCE_RATIO * n.distance:
            good.append([m])
    return good


def get_img(img_name):
    return cv2.imread(DATA_DIR + img_name)



if __name__ == "__main__":
    for image_pair in IMAGES:
        img_0 = get_img(image_pair[0])
        img_1 = get_img(image_pair[1])

        kp_0, des_0 = find_img_keypoint(img_0)
        kp_1, des_1 = find_img_keypoint(img_1)

        good_matches = find_good_matches(des_0, des_1)
        img = cv2.drawMatchesKnn(img_0, kp_0, img_1, kp_1, good_matches, None, flags=2)

        plt.imshow(img)
        plt.show()


