import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal

IMAGES = [["Mesona1.JPG", "Mesona2.JPG"], ["Statue1.bmp", "Statue2.bmp"]]
INTRINSIC = ["Mesona_calib.txt", "Statue_calib.txt"]

GOOD_MATCH_K = 2
GOOD_DISTANCE_RATIO = 0.3
RANSAC_THRESHOLD = 0.6


def get_img(img_name):
    return cv2.imread(img_name)


def cv2_img_show(img):
    cv2.imshow("img", np.array(img, dtype = np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_calib_data(calib_file):
    """
    For case 1, only calculate K here.
    """
    K = []
    R = []
    t = []
    
    fp = open(calib_file, 'r')
    line = fp.readline()
    if line != "calibration parameters": # Don't need calibrate
        for _ in range(3): # K
            line = fp.readline()
            row_of_K = [float(val) for val in line.split(' ')]
            K.append(row_of_K)
        line = fp.readline()
        for _ in range(3): # R
            line = fp.readline()
            row_of_R = [float(val) for val in line.split(' ')]
            R.append(row_of_R)
        line = fp.readline()
        for _ in range(1): # t
            line = fp.readline()
            row_of_t = [float(val) for val in line.split(' ')]
            t.append(row_of_t)
    return np.array(K), np.array(R), np.array(t).T


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
    x1_padding = np.ones((x1.shape[0], 3))
    x1_padding[:, :-1] = x1
    x2_padding = np.ones((x1.shape[0], 3))
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


def get_pinhole_intrinsic_params():
    K = []
    with open(calibration_file_dir + '/camera_observatory.txt') as f:
        lines = f.readlines()
        calib_info = [float(val) for val in lines[0].split(' ')]
        row1 = [calib_info[0], 0, calib_info[2]]
        row2 = [0, calib_info[1], calib_info[3]]
        row3 = [0, 0, 1]

        K.append(row1)
        K.append(row2)
        K.append(row3)
    
    return K


def compute_essential_candidate(f, K):
    E = K.T @ f @ K
    U, S, V = np.linalg.svd(E)
    S = np.diag(S)
    
    m = (S[1, 1] + S[2, 2]) / 2
    E = U @ np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]]) @ V.T
    
    U, S, V = np.linalg.svd(E)
    S = np.diag(S)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    P1 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # P2 chooses
    P2_1 = np.zeros((3, 4))
    P2_1[:, :-1] = U @ W @ V.T
    P2_1[:, -1] = U[2]

    P2_2 = np.zeros((3, 4))
    P2_2[:, :-1] = U @ W @ V.T
    P2_2[:, -1] = (-1)*U[2]

    P2_3 = np.zeros((3, 4))
    P2_3[:, :-1] = U @ W.T @ V.T
    P2_3[:, -1] = U[2]

    P2_4 = np.zeros((3, 4))
    P2_4[:, :-1] = U @ W.T @ V.T
    P2_4[:, -1] = (-1)*U[2]

    P2_chooses = np.array([P2_1, P2_2, P2_3, P2_4])    

    return P1, P2_chooses


def triangulation(x1, x2, P2_chooses, R, t):
    ### add 1 for homogeneous ###
    x1_extend = np.ones((x1.shape[0], 3))
    x1_extend[:, :-1] = x1
    x2_extend = np.ones((x1.shape[0], 3))
    x2_extend[:, :-1] = x2

    ### find best solution between each P2_chooses ###
    cnt_of_front_points = np.zeros(4)
    for i, P in enumerate(P2_chooses):
        ### Create matrix A ###
        for point_idx in range(x1_extend.shape[0]):
            u_1, v_1 = x1_extend[point_idx, 0], x1_extend[point_idx, 1]
            u_2, v_2 = x2_extend[point_idx, 0], x2_extend[point_idx, 1]
            # print(u_1.shape)
            P_1 = P[0].reshape((P[0].shape[0], 1))
            P_2 = P[1].reshape((P[1].shape[0], 1))
            P_3 = P[2].reshape((P[2].shape[0], 1))
            print(P_1.shape)
            # print(u_1 @ P_0.T)
            A = [u_1 * P_3.T - P_1.T, v_1 * P_3.T - P_2.T, u_2 * P_3.T - P_1.T, v_2 * P_3.T - P_2.T]
            A = np.array(A)
            A = A.reshape((4, 4))
            print('A: ', A.shape)

            ### calculate X ###
            U, S, V = np.linalg.svd(A)
            X = V[:, -1]
            print(X.shape)
            print(X)

            ### Count in-front-of-camera point ###
            C = (-1) * R.T @ t
            print(C.shape)
            view_direction = R[2, :].T
            if (X-C) @ view_direction > 0:
                # in front!
                cnt_of_front_points[i] += 1
    print('cnt: ', cnt_of_front_points)


def sfm(img_0, img_1, K, R, t):
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

    ### 3. Draw interest points and the corresponding epipolar lines ###

    ### 4. Get 4 possible solutions of essential matrix from f ###
    P1, P2_chooses = compute_essential_candidate(f, K)
    print('P2 chooses: ', P2_chooses)

    ### 5. Find out most appropriate solution ###
    P2 = triangulation(match_points_0, match_points_1, P2_chooses, R, t)


if __name__ == "__main__":
    for i, image_pair in enumerate(IMAGES):
        img1 = get_img(image_pair[0])
        img2 = get_img(image_pair[1])
        K, R, t = get_calib_data(INTRINSIC[i])
        
        
        sfm(img1, img2, K, R, t)
        exit(0)

    