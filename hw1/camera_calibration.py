import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
    # shape(corners) = (49, 1, 2)
    corners = corners.reshape(49,2)
    # corners = np.flip(np.flip(corners.reshape(49,2), 0).reshape(7,7,2).transpose(1,0,2), 0).reshape(49,2)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
#         plt.imshow(img)
#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
# img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
# Vr = np.array(rvecs)
# Tr = np.array(tvecs)
# extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
Write your code here
"""
def append_p(P, obj, img):
    P.append([obj[0], obj[1], 1, 0, 0, 0, -img[0]*obj[0], -img[0]*obj[1], -img[0]])
    P.append([0, 0, 0, obj[0], obj[1], 1, -img[1]*obj[0], -img[1]*obj[1], -img[1]])

def get_v(Hi, Hj):
    return np.array([Hi[0]*Hj[0],
                     Hi[0]*Hj[1]+Hi[1]*Hj[0],
                     Hi[1]*Hj[1],
                     Hi[2]*Hj[0]+Hi[0]*Hj[2],
                     Hi[2]*Hj[1]+Hi[1]*Hj[2],
                     Hi[2]*Hj[2]])

V = []
# for i in range(np.shape(objpoints)[0]):
for i, fname in enumerate(images):
    print(i)
    P = []
    # print("imgpoints is ",imgpoints[i])
    # print("objpoints is ",objpoints[i])
    for j in range(corner_x*corner_y):
        append_p(P, objpoints[i][j], imgpoints[i][j])
    P = np.array(P)
    print("P shape is ",P.shape)
    u, s, vh = np.linalg.svd(P)
    print("P 's singular",s)
    H = vh[np.argmin(s)]
    H = np.array([[H[0], H[1], H[2]],
                  [H[3], H[4], H[5]],
                  [H[6], H[7], H[8]]])
    """ This part doesn't work anyway """
    # ones = np.ones((corner_x*corner_y,1), np.float32)
    # imgvec = np.concatenate((imgpoints[i],ones),axis=1)
    # print("imgvec=",imgvec) 
    # # [U,V,1] is objpoints
    # objvec = np.concatenate((objpoints[i],ones),axis=1)
    # print("objvec=",objvec) 
    # # get H by simply using [u,v] and [U,V]
    # H = imgvec.dot(np.linalg.inv(objvec))
    print("H=",H)
    v12 = get_v(H[:,0], H[:,1])
    v11 = get_v(H[:,0], H[:,0])
    v22 = get_v(H[:,1], H[:,1])
    V.append(v12)
    V.append(v11-v22)


V = np.array(V)
print("V shape = ",np.shape(V))
print(V)
u, s, vh = np.linalg.svd(V)
# print(V.T.dot(V))
# u, s, vh = np.linalg.svd(V.T.dot(V))
print("u, s, vh")
print("s=",s)
print("vh=",vh)
b = vh[np.argmin(s)]
print("[SVD] b=",b)

w,v = np.linalg.eig(np.dot(V.T,V))
print("w,v")
print(w,"\n",v)

B = np.array([[b[0], b[1], b[3]],
            [b[1], b[2], b[4]],
            [b[3], b[4], b[5]]])

# try:
#     L = np.linalg.cholesky(B)
#     print("B can do cholesky")
#     # print(L)
# except:
#     w, v = np.linalg.eig(B)
#     print(w)
#     print("B failed")
#     # L = np.linalg.cholesky(-1*B)
#     raise SystemExit(0)

# # B = K.inv.T * K.inv
# # B = L * L.H
# # H = L.inv
# K = np.linalg.inv(L.T)
# print("B=",B,"K=",K)
# print("K.inv.T dot K.inv = ",np.dot(np.linalg.inv(K.T),np.linalg.inv(K)))
# # K[0,1] = 0.0
# K = K/K[2,2]

# Use Zhang's method instead of cholesky
# A = [[alpha, gama, u0],[0,beta,v0],[0,0,1]]
v0 = (B[0][1]*B[0][2] - B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]*B[0][1])
landa = B[2][2] - (B[0][2]**2 + v0*(B[0][1]*B[0][2] - B[0][0]*B[1][2]))/B[0][0]
alpha = np.sqrt(landa/B[0][0])
beta = np.sqrt((landa*B[0][0])/(B[0][0]*B[1][1] - B[0][1]*B[0][1]))
gama = -(B[0][1]*alpha*alpha*beta)/landa
u0 = (gama*v0)/beta - (B[0][2]*alpha*alpha)/landa
K = np.array([[alpha,gama,u0],[0,beta,v0],[0,0,1]])
print("K=",K)
mtx = K
Kinv = np.linalg.inv(K)
# r1 = lambda * K.inv * h1
# r2 = lambda * K.inv * h2
# r3 = r1 x r2
# t = lambda * K.inv h3
# lambda = 1/||K.inv * h1||
extrinsics = []
for i, fname in enumerate(images):
    print(i)
    P = []
    for j in range(corner_x*corner_y):
        append_p(P, objpoints[i][j], imgpoints[i][j])
    P = np.array(P)
    print(P.shape)
    u, s, vh = np.linalg.svd(P)
    H = vh[np.argmin(s)]
    # H = np.array([[H[0], H[1], H[2]],
    #               [H[3], H[4], H[5]],
    #               [H[6], H[7], H[8]]])
    h1 = np.array([H[0], H[3], H[6]])
    h2 = np.array([H[1], H[4], H[7]])
    h3 = np.array([H[2], H[5], H[8]])
    landa1 = 1/(np.linalg.norm(Kinv.dot(h1)))
    landa2 = 1/(np.linalg.norm(Kinv.dot(h2)))
    landa = np.sqrt(landa1*landa2)
    print("landa=",landa)
    r1 = landa * Kinv.dot(h1)
    r2 = landa * Kinv.dot(h2)
    r3 = np.cross(r1,r2)
    t = landa * Kinv.dot(h3)
    # if r1[0] < 0:
    #     extrinsics.append(-1*np.array([r1,r2,r3,t]).T)
    # else:
    #     extrinsics.append(np.array([r1,r2,r3,t]).T)
    extrinsics.append(np.array([r1,r2,r3,t]).T)
    print(r1,r2,r3,t)
    print(np.array([r1,r2,r3,t]).T)
extrinsics = np.array(extrinsics)
print("my extrinsics \n",extrinsics)
for idx in range(extrinsics.shape[0]):
    R = extrinsics[idx,0:3,0:3]
    T = extrinsics[idx,0:3,3]
print("R=",R)
print("T=",T)


# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, False)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""

# TA
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
print("TA's extrintics \n",extrinsics)
for idx in range(extrinsics.shape[0]):
    R, _ = cv2.Rodrigues(extrinsics[idx,0:3])
    T = extrinsics[idx,3:6]
print("R=",R)
print("T=",T)

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = mtx
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, False)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()