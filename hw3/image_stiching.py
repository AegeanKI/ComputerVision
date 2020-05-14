import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import geometric_transform

DATA_DIR = 'data/'
IMAGES = [["2.jpg", "1.jpg"], ["hill2.JPG", "hill1.JPG"], ["S2.jpg", "S1.jpg"]]
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

def transform_to_correspond_points(kp_0, kp_1, good_matches):
    correspond_match_points = []
    for m in good_matches:
        point_0 = kp_0[m.queryIdx].pt
        point_1 = kp_1[m.trainIdx].pt
        correspond_match_points.append([point_0, point_1])
    return correspond_match_points

def geometricDistance(p0, p1, h):

    extend_p0 = np.array([p0[0], p0[1], 1])
    estimate_p1 = np.dot(h, extend_p0)
    estimate_p1 = estimate_p1 / estimate_p1[-1]
    
    extend_p1 = np.array([p1[0], p1[1], 1])
    error = extend_p1 - estimate_p1
    return np.linalg.norm(error)

def append_p(P, obj, img):
    P.append([obj[0], obj[1], 1, 0, 0, 0, -img[0]*obj[0], -img[0]*obj[1], -img[0]])
    P.append([0, 0, 0, obj[0], obj[1], 1, -img[1]*obj[0], -img[1]*obj[1], -img[1]])

def calculate_homography(points_0, points_1):
    P = []
    for j in range(points_0.shape[0]):
        append_p(P, points_0[j], points_1[j])
    P = np.array(P)
    u, s, vh = np.linalg.svd(P)
    H = vh[-1]
    H = H / H[-1]
    H = np.array([[H[0], H[1], H[2]],
                  [H[3], H[4], H[5]],
                  [H[6], H[7], H[8]]])
    return H

def RANSAC_homography(match_points_0, match_points_1):
    max_inliers = []
    max_inlier_h = None
    for i in range(100):
        idx = random.sample(range(0, len(match_points_0)), 5)
        h = calculate_homography(match_points_0[idx], match_points_1[idx])

        inliers = []
        for p0, p1 in zip(match_points_0, match_points_1):
            d = geometricDistance(p0, p1, h)
            if d < 5:
                inliers.append([p0, p1])
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            max_inlier_h = h

    return max_inlier_h, np.array(max_inliers)

def warp(img_0, img_1, H):
#     result = cv2.warpPerspective(img_0, H, (img_0.shape[1] + img_1.shape[1], img_0.shape[0]))
    result = warpPerspective(img_0, H, (img_0.shape[1] + img_1.shape[1], img_0.shape[0]))
    result[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    return result

def get_img(img_name):
    return cv2.imread(DATA_DIR + img_name)

def cv2_img_show(img):
    cv2.imshow("img", np.array(img, dtype = np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def warpPerspective(img, H, dsize):
    result_width, result_height = dsize
    img_width, img_height = img.shape[0], img.shape[1]

    img_swapped = np.swapaxes(img, 0, 1)
    H_inv = np.linalg.inv(H)
    result = np.zeros((result_width, result_height, 3))
    for x in range(result_width):
        for y in range(result_height):
            origin_p = np.dot(H_inv, [x, y, 1])
            x_origin, y_origin, _ = (origin_p / origin_p[2] + 0.5).astype(int)
            if x_origin >= 0 and x_origin < img.shape[1]:
                if y_origin >= 0 and y_origin < img.shape[0]:
                    result[x, y] = img_swapped[x_origin, y_origin]
    return np.swapaxes(result, 0, 1)


if __name__ == "__main__":
    for image_pair in IMAGES:
        img_0 = get_img(image_pair[0])
        img_1 = get_img(image_pair[1])

        kp_0, des_0 = find_img_keypoint(img_0)
        kp_1, des_1 = find_img_keypoint(img_1)

        good_matches, good_matches_for_img_show = find_good_matches(des_0, des_1)
#         img = cv2.drawMatchesKnn(img_0, kp_0, img_1, kp_1, good_matches_for_img_show, None, flags=2)
#         cv2_img_show(img)

        match_points_0 = np.float32([kp_0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2) 
        match_points_1 = np.float32([kp_1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        h, inliers = RANSAC_homography(match_points_0, match_points_1)
#         img = cv2.drawMatchesKnn(img_0, kp_0, img_1, kp_1, good_matches_for_img_show, np.array(inliers), flags=2)
#         cv2_img_show(img)
        img = warp(img_0, img_1, h)
        # img = warp(img_1, img_0, h)
        cv2_img_show(img)
        exit(0)

#         break
    # cv2.waitKey(0)




