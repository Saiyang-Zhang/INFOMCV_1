import cv2
import os
import numpy as np
import glob


# termination criteria, maximum number of loops = 30 and maximum error tolerance = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
w = 9
h = 6

# checkerboard points in the world coordinate system, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),
# with the z-coordinate removed, recorded as a two-dimensional matrix
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp * 23    #grid length = 23mm

# Store the world coordinates and image coordinates of the checkerboard grid
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./CameraRoll/*.jpg')

#i=0
for fname in images:
    img = cv2.imread(fname)
    h1, w1 = img.shape[0], img.shape[1]
    u, v = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # If found, save the information
    if ret == True:
        #print("i", i)
        #i += 1

        #Finding sub-pixel corners based on the original corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners', img)
        cv2.waitKey(200)
    cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print("ret:", ret)
# print("mtx:\n", mtx)    # 内参数矩阵
# print("dist:\n", dist)  # 畸变系数   distortion coefficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs:\n", rvecs)    # 旋转向量,外参数
# print("tvecs:\n", tvecs )   # 平移向量,外参数
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u,v), 0, (u,v))
# print("newcameramtx", newcameramtx) # 外参数

# Capture picture using webcam
camera=cv2.VideoCapture(0)
n = 0
m = 100
while True:
    (grabbed, frame) = camera.read()
    h1,w1 = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u,v), 0, (u,v))

    # undistort
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # remapping, same result as undistort
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w1,h1), 5)
    # dst2 = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w1, h1 = roi
    dst1 = dst1[y:y+h1, x:x+w1]
    cv2.imshow("NewImage", dst1)
    k=cv2.waitKey(1)
    if k==27:   #press Esc to quit
        break
    elif k == ord('s'):   #press s to save the image
        cv2.imwrite('./UndistortImage/' + str(n) + '.jpg', dst1)
        n += 1

camera.release()
cv2.destroyAllWindows()


