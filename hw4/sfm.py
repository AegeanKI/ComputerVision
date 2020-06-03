import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
from scipy.spatial import Delaunay

IMAGES = [["Mesona1.JPG", "Mesona2.JPG"], ["Statue1.bmp", "Statue2.bmp"]]
INTRINSIC = ["Mesona_calib.txt", "Statue_calib.txt"]

GOOD_MATCH_K = 2
GOOD_DISTANCE_RATIO = 0.8
RANSAC_THRESHOLD = 0.6

######################### Get Data ###################################

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


##################### Find matching points ###############################

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

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    num_good_matches = int(len(good_matches) * 0.25)
    good_matches = good_matches[:num_good_matches]
    return good_matches, good_matches_for_img_show


def find_img_keypoint(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des


# def find_good_matches(des_0, des_1):
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des_0, des_1, k=GOOD_MATCH_K)
#     good_matches_for_img_show = []
#     good_matches = []
#     for m, n in matches:
#         if m.distance < GOOD_DISTANCE_RATIO * n.distance:
#             good_matches_for_img_show.append([m])
#             good_matches.append(m)
#     return good_matches, good_matches_for_img_show


##################### Compute RANSAC (fundamental) ###############################

def geometricDistance(p0, p1, f, shape):
    disparity = abs(p1.reshape(3,1).T @ f @ p0.reshape(3,1))[0,0]
    disparity = abs(p1.T @ f @ p0)
    return disparity

def append_p(P, obj, img):
    P.append([obj[0], obj[1], 1, 0, 0, 0, -img[0]*obj[0], -img[0]*obj[1], -img[0]])
    P.append([0, 0, 0, obj[0], obj[1], 1, -img[1]*obj[0], -img[1]*obj[1], -img[1]])

def GetRowOf_F(a, b):
    return [a[0]*b[0], a[1]*b[0], b[0], a[0]*b[1], a[1]*b[1], b[1], a[0], a[1], 1 ]


def compute_fundamental(x1,x2):
    arr = []
    for i in range(len(x1)):
        arr.append(GetRowOf_F(x1[i], x2[i]))

    U, S, VH = np.linalg.svd(np.array(arr))
    F = np.reshape(VH[-1], (3, 3))
    
    U, D, VH = np.linalg.svd(F)
    D[2] = 0
    F = U @ np.diag(D) @ VH
    F /= F[-1, -1]

    return F

def compute_normalize_point(pt, matrix):
    corr = []
    for i in pt:
        cood = np.atleast_2d(np.array([i[0], i[1], 1])).T
        cood = matrix @ cood
        cood = cood.flatten()
        corr.append(cood)
    return np.array(corr)


def RANSAC_fundamental(match_points_0, match_points_1, shape):
    max_inliers = []
    # max_inlier_h = None
    max_inlier_f = None
    T1 = np.array([[2/shape[1], 0, -1], 
                   [0, 2/shape[0], -1],
                   [0, 0, 1]])
    print("match_points_0.shape:")

    print(T1.shape)
    normal_p0 = compute_normalize_point(match_points_0, T1)
    normal_p1 = compute_normalize_point(match_points_1, T1)

    for i in range(1000):
        idx = np.random.randint(0, len(normal_p0), 60)
        f = compute_fundamental(normal_p0[idx], normal_p1[idx])

        inliers = []
        for p0, p1 in zip(normal_p0, normal_p1):
            d = geometricDistance(p0, p1, f, shape)
#             print('d:', d)
            if d < 0.01:
                inliers.append([p0, p1])
        if len(inliers) > len(max_inliers):
            print("inliner length: {}".format(len(inliers)))
            max_inliers = inliers
            # max_inlier_h = h
            max_inlier_f = f
    max_inlier_f = T1.T @ max_inlier_f @ T1
    max_inlier_f /= max_inlier_f[-1, -1]

    return max_inlier_f, np.array(max_inliers)


################# Compute Solution of Essential Matrix ########################

def compute_essential_candidate(f, K):
    E = K.T @ f @ K
    U, S, VH = np.linalg.svd(E)
    S = np.diag(S)
    
    m = (S[1, 1] + S[2, 2]) / 2
    E = U @ np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]]) @ VH
    
    U, S, VH = np.linalg.svd(E)
    S = np.diag(S)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    P1 = K @ np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
#     P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # P2 chooses
    P2_1 = np.zeros((3, 4))
    P2_1[:, :-1] = U @ W @ VH
    P2_1[:, -1] = U[2]

    P2_2 = np.zeros((3, 4))
    P2_2[:, :-1] = U @ W @ VH
    P2_2[:, -1] = (-1)*U[2]

    P2_3 = np.zeros((3, 4))
    P2_3[:, :-1] = U @ W.T @ VH
    P2_3[:, -1] = U[2]

    P2_4 = np.zeros((3, 4))
    P2_4[:, :-1] = U @ W.T @ VH
    P2_4[:, -1] = (-1)*U[2]

    P2_chooses = np.array([P2_1, P2_2, P2_3, P2_4])    

    return P1, P2_chooses


def find_best_solution(x1, x2, P2_chooses):
    ### add 1 for homogeneous ###
    x1_extend = np.ones((x1.shape[0], 3))
    x1_extend[:, :-1] = x1
    x2_extend = np.ones((x1.shape[0], 3))
    x2_extend[:, :-1] = x2
    
    ### find best solution between each P2_chooses ###
    cnt_of_front_points = np.zeros(4)
    for i, P in enumerate(P2_chooses):
        R = P[:, :-1]
        t = P[:,-1].reshape(3,1)
        ### Create matrix A ###
        for point_idx in range(x1_extend.shape[0]): # for each [u,v] pair
            u_1, v_1 = x1_extend[point_idx, 0], x1_extend[point_idx, 1]
            u_2, v_2 = x2_extend[point_idx, 0], x2_extend[point_idx, 1]
            # print(u_1.shape)
            P_1 = P[0].reshape((P[0].shape[0], 1))
            P_2 = P[1].reshape((P[1].shape[0], 1))
            P_3 = P[2].reshape((P[2].shape[0], 1))
            # print(P_1.shape)
            # print(u_1 @ P_0.T)
            A = [u_1 * P_3.T - P_1.T,
                 v_1 * P_3.T - P_2.T,
                 u_2 * P_3.T - P_1.T,
                 v_2 * P_3.T - P_2.T]
            A = np.array(A)
            A = A.reshape((4, 4))
            # print('A: ', A.shape)

            ### calculate X ###
            U, S, VH = np.linalg.svd(A)
            X = VH[-1]
            X = X / X[-1]
            X = X[:-1].reshape((3, 1))
            
            ### Count in-front-of-camera point ###
            C = (-1) * R.T @ t            
            view_direction = (R[2, :].T).reshape((3, 1))
            if ((X-C).T @ view_direction).item() > 0:
                # in front!
#                 print('inner product: ', (X-C).T @ view_direction)
                cnt_of_front_points[i] += 1
    print('cnt: ', cnt_of_front_points)
    
    max_p2_idx = np.argmax(cnt_of_front_points)
    return P2_chooses[max_p2_idx]


##################### Triangulation ############################


def Triangulation(x1, x2, P1, P2):
    X = []
    p1 = P1[0, :]
    p2 = P1[1, :]
    p3 = P1[2, :]
    p1p = P2[0, :]
    p2p = P2[1, :]
    p3p = P2[2, :]

    for i in range(len(x1)):
        u = x1[i, 0]
        v = x1[i, 1]
        up = x2[i, 0]
        vp = x2[i, 1]
        A = np.array([u  * p3 - p1,
                      v  * p3 - p2,
                      up * p3p - p1p,
                      vp * p3p - p2p])

        U, S, VH = np.linalg.svd(A)
        X.append((VH[-1] / VH[-1, -1])[:3])

    return np.array(X)


##################### Main Driven Function ###############################
def drawlines(img1,img2,lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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

#     print(gray_0.shape)
    print("match_points_0")
    print(match_points_0)

    ### 2. Estimate fundamental matrix ###
    f, inliers = RANSAC_fundamental(match_points_0, match_points_1, gray_0.shape)
    print("f:")
    print(f)

    ### 3. Drlines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = cv2.computeCorrespondEpilines(match_points_1.reshape(-1,1,2), 2, f)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(gray_0, gray_1, lines1, match_points_0, match_points_1)
    plt.imshow(img5)
    plt.show()

    ### 4. Get 4 possible solutions of essential matrix from f ###
    P1, P2_chooses = compute_essential_candidate(f, K)
    print('P2 chooses: ', P2_chooses)

    ### 5. Find out most appropriate solution ###
    P2 = find_best_solution(match_points_0, match_points_1, P2_chooses)
    print("P2:")
    print(P2)

    ### 6. Triangulation ###
    X = Triangulation(match_points_0, match_points_1, P1, P2)
    print('X: ', X.shape)
    print('X: ', X)

if __name__ == "__main__":
    for i, image_pair in enumerate(IMAGES):
        img1 = get_img(image_pair[0])
        img2 = get_img(image_pair[1])
        K, R, t = get_calib_data(INTRINSIC[i])
        
        
        sfm(img1, img2, K, R, t)
        exit(0)

    
