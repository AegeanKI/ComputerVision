import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal

IMAGES = [["Mesona1.JPG", "Mesona2.JPG"], ["Statue1.bmp", "Statue2.bmp"]]
INTRINSIC = [None, "Statue_calib.txt"]

GOOD_MATCH_K = 2
GOOD_DISTANCE_RATIO = 0.3
RANSAC_THRESHOLD = 0.6


def find_img_keypoint(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def find_good_matches(des_0, des_1):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_0, des_1, k=GOOD_MATCH_K)
    good_matches_for_img_show = []
    good_matches = []
    for m, n in matches:
        if m.distance < GOOD_DISTANCE_RATIO * n.distance:
            good_matches_for_img_show.append([m])
            good_matches.append(m)
    return good_matches, good_matches_for_img_show


def get_img(img_name):
    return cv2.imread(img_name)


def cv2_img_show(img):
    cv2.imshow("img", np.array(img, dtype = np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def correspondence(gray_0, gray_1):
    """
    KLT
    """
    ### finding the good features ###
    features = cv2.goodFeaturesToTrack(gray_0 # Input image
    ,10000 # max corners
    ,0.01 # lambda 1 (quality)
    ,10 # lambda 2 (quality)
    )
    feature = np.int0(features)
    # First Derivative in X direction
    Ix = signal.convolve2d(gray_0,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(gray_1,[[-0.25,0.25],[-0.25,0.25]],'same')
    # First Derivative in Y direction
    Iy = signal.convolve2d(gray_0,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(gray_1,[[-0.25,-0.25],[0.25,0.25]],'same')
    # First Derivative in XY direction
    It = signal.convolve2d(gray_0,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(gray_1,[[-0.25,-0.25],[-0.25,-0.25]],'same')
    

    ### Use 2nd moment matrix & difference across frames to fine displacement ###
    #creating the u and v vector
    u = v = np.nan*np.ones(S)

    # Calculating the u and v arrays for the good features obtained n the previous step.
    for l in feature:
        j,i = l.ravel()
        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.

        IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]]) #The x-component of the gradient vector
        IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]]) #The Y-component of the gradient vector
        IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]]) #The XY-component of the gradient vector

        # Using the minimum least squares solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK)) # transpose of A
        LK = np.array(np.matrix.transpose(LK)) 

        A1 = np.dot(LK_T,LK) #Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2,LK_T)

        (u[i,j],v[i,j]) = np.dot(A3,IT) # we have the vectors with minimized square error
    return u, v


def sfm(img1, img2, intrinsic):
    # 0. Calibration

    ### 1. Find correspondence across images ###
    gray_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    u, v = correspondence(gray_0, gray_1)

    


if __name__ == "__main__":
    for i, image_pair in enumerate(IMAGES):
        img1 = get_img(image_pair[0])
        img2 = get_img(image_pair[1])
        intrinsic_matrix = INTRINSIC[i]

        sfm(img1, img2, intrinsic_matrix)
        exit(0)

    