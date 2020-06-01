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

def KLT(gray_0, gray_1):
    """
    KLT
    """
    S = np.shape(gray_0)
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

def compute_fundamental(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))
    
    return F/F[2,2]


def sfm(img_0, img_1, intrinsic):
    # 0. Calibration

    ### 1. Find correspondence across images ###
    gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    
    kp_0, des_0 = find_img_keypoint(img_0)
    kp_1, des_1 = find_img_keypoint(img_1)

    good_matches, good_matches_for_img_show = find_good_matches(des_0, des_1)
    # img = cv2.drawMatchesKnn(img_0, kp_0, img_1, kp_1, good_matches_for_img_show, None, flags=2)
    # cv2_img_show(img)

    match_points_0 = np.float32([kp_0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) 
    match_points_1 = np.float32([kp_1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    print(gray_0.shape)
    print(match_points_0.shape)

    ### 2. Estimate fundamental matrix ###

    u, v = KLT(gray_0, gray_1)


    


if __name__ == "__main__":
    for i, image_pair in enumerate(IMAGES):
        img1 = get_img(image_pair[0])
        img2 = get_img(image_pair[1])
        intrinsic_matrix = INTRINSIC[i]

        sfm(img1, img2, intrinsic_matrix)
        exit(0)

    