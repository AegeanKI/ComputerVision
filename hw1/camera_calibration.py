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
#     corners = corners.reshape(49,2)
    corners = np.flip(np.flip(corners.reshape(49,2), 0).reshape(7,7,2).transpose(1,0,2), 0).reshape(49,2)

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
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
Write your code here
"""
def get_v(H, i, j):
#     return np.array([H[0][i]*H[0][j],
#                     H[0][i]*H[1][j]+H[1][i]*H[0][j],
#                     H[1][i]*H[1][j],
#                     H[2][i]*H[0][j]+H[0][i]*H[2][j],
#                     H[2][i]*H[1][j]+H[1][i]*H[2][j],
#                     H[2][i]*H[2][j]])
    return np.array([H[i][0]*H[j][0],
                    H[i][0]*H[j][1]+H[i][1]*H[j][0],
                    H[i][1]*H[j][1],
                    H[i][2]*H[j][0]+H[i][0]*H[j][2],
                    H[i][2]*H[j][1]+H[i][1]*H[j][2],
                    H[i][2]*H[j][2]])

V = []
for i in range(np.shape(objpoints)[0]):
    print(i)
    objmatrix = objpoints[i]
    objmatrix[:,-1] = np.ones(corner_x*corner_y, np.float32)
    print(objmatrix.shape)
    
    imgmatrix = np.ones((corner_x*corner_y, 3), np.float32)
    imgmatrix[:,:-1] = imgpoints[i]
    print(imgmatrix.shape)

    # 0, 1, 2 use (no T and down) or (T and up)  can let u5 do cholesky
    # 0, 1, 2, 3 use (no T and up) or (T and down) can let u4 do cholesky
    H = np.linalg.inv(objmatrix.T.dot(objmatrix)).dot(objmatrix.T).dot(imgmatrix).T

#     # normalize
#     lengths = np.linalg.norm(H, axis=-1)
#     H[lengths > 0] = H[lengths > 0] / lengths[lengths > 0][:, np.newaxis]
    print(H)
    v12 = get_v(H, 0, 1)
    v11 = get_v(H, 0, 0)
    v22 = get_v(H, 1, 1)
    V.append(v12)
    V.append(v11-v12)
    



V = np.array(V)
print(np.shape(V))
u, s, vh = np.linalg.svd(V.dot(V))
# print("u, s, vh")
print(u)
print(s)
# min_eigenvalue_index = list(s).index(min(s))
# b = u[min_eigenvalue_index]

for i in range(0, s.size):
    b = u[i]
    print(b)
    B = np.array([[b[0], b[1], b[3]],
                [b[1], b[2], b[4]],
                [b[3], b[4], b[5]]])
    try:
        L = np.linalg.cholesky(B)
        print("u{}.can do cholesky".format(i))
    except:
        print("u{} failed".format(i))

for i in range(0, s.size):
    b = vh[i]
    print(b)
    B = np.array([[b[0], b[1], b[3]],
                [b[1], b[2], b[4]],
                [b[3], b[4], b[5]]])
    try:
        L = np.linalg.cholesky(B)
        print("vh{}.can do cholesky".format(i))
    except:
        print("vh{} failed".format(i))
raise SystemExit(0)





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
                                                board_height, square_size, True)

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
# plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
