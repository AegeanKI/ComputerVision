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


def geometricDistance(p0, p1, f):

    extend_p0 = np.array([p0[0], p0[1], 1])
    # estimate_p1 = np.dot(h, extend_p0)
    # estimate_p1 = estimate_p1 / estimate_p1[-1]
    
    extend_p1 = np.array([p1[0], p1[1], 1])
    # error = extend_p1 - estimate_p1
    # print('p0:', extend_p0.shape)
    # print('p1:', extend_p1.shape)
    # print('f:', f.shape)
    # # exit(0)
    disparity = np.linalg.norm((extend_p1 @ f) @ extend_p0)
    # return np.linalg.norm(error)
    return disparity

def append_p(P, obj, img):
    P.append([obj[0], obj[1], 1, 0, 0, 0, -img[0]*obj[0], -img[0]*obj[1], -img[0]])
    P.append([0, 0, 0, obj[0], obj[1], 1, -img[1]*obj[0], -img[1]*obj[1], -img[1]])


def compute_fundamental(x1,x2):
    """
    Computes the fundamental matrix from corresponding points 
    (x1,x2 3*n arrays) using the 8 point algorithm.
    Each row in the A matrix below is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """    
    n = x1.shape[1]
       
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))

    return F/F[2,2]


def compute_fundamental_normalized(x1, x2, shape):
    """
    Computes the fundamental matrix from corresponding points 
    (x1,x2 3*n arrays) using the normalized 8 point algorithm. 
    """
    # n = x1.shape[1]

    # extend x1, x2
    x1_padding = np.ones((37, 3))
    x1_padding[:, :-1] = x1
    x2_padding = np.ones((37, 3))
    x2_padding[:, :-1] = x2

    # normalize image coordinates
    # x1 = x1 / x1[2]
    # mean_1 = np.mean(x1[:2],axis=1)
    # S1 = np.sqrt(2) / np.std(x1[:2])
    # T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    # x1 = np.dot(T1,x1)
    T1 = np.array([[2/shape[0], 0, -1], [0, 2/shape[1], -1], [0, 0, 1]])
    x1_padding = np.dot(T1, x1_padding.T)
    x2_padding = np.dot(T1, x2_padding.T)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1_padding,x2_padding)

    # reverse normalization
    F = np.dot(T1.T, np.dot(F, T1))

    return F/F[2,2]


def RANSAC_fundamental(match_points_0, match_points_1, shape):
    max_inliers = []
    # max_inlier_h = None
    max_inlier_f = None
    for i in range(100):
        idx = random.sample(range(0, len(match_points_0)), 5)
        # h = calculate_homography(match_points_0[idx], match_points_1[idx])
        # f = compute_fundamental(match_points_0, match_points_1)
        f = compute_fundamental_normalized(match_points_0, match_points_1, shape)
        print('f:', f)

        inliers = []
        for p0, p1 in zip(match_points_0, match_points_1):
            d = geometricDistance(p0, p1, f)
            print('d:', d)
            if d < 5:
                inliers.append([p0, p1])
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            # max_inlier_h = h
            max_inlier_f = f

    return max_inlier_f, np.array(max_inliers)


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
    f, inliers = RANSAC_fundamental(match_points_0, match_points_1, gray_0.shape)
    print(f)


    


if __name__ == "__main__":
    for i, image_pair in enumerate(IMAGES):
        img1 = get_img(image_pair[0])
        img2 = get_img(image_pair[1])
        intrinsic_matrix = INTRINSIC[i]

        sfm(img1, img2, intrinsic_matrix)
        exit(0)

    