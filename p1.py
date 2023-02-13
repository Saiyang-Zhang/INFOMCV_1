import cv2
import numpy as np
import glob

# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the Shell
        print(x, ' ', y)
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(params, str(x)+','+str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('findCorners', params)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the Shell
        print(x, ' ', y)
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = params[y, x, 0]
        g = params[y, x, 1]
        r = params[y, x, 2]
        cv2.putText(params, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('findCorners', params)

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-2)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),2)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),2)
    return img

# offline phrase
def offline():
    # termination criteria, maximum number of loops = 30 and maximum error tolerance = 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w = 9
    h = 6

    # checkerboard points in the world coordinate system, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),
    # with the z-coordinate removed, recorded as a two-dimensional matrix
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    #objp = objp * 23    #grid length = 23mm

    # Store the world coordinates and image coordinates of the checkerboard grid
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./CameraRoll/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        #h1, w1 = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        # If found, save the information
        if ret == True:
            #Finding sub-pixel corners based on the original corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

        else:
            # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('findCorners', 640, 480)
            # cv2.imshow('findCorners', img)
            # cv2.setMouseCallback('findCorners', click_event, img)
            # corners2 = 线性插值？
            # objpoints.append(objp)
            # imgpoints.append(corners2)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()
            print("a")

    print("imgpoints:", imgpoints)
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print("ret:", ret)
    # print("mtx:\n", mtx)    # 内参数矩阵
    # print("dist:\n", dist)  # 畸变系数   distortion coefficients = (k_1,k_2,p_1,p_2,k_3)
    # print("rvecs:\n", rvecs)    # 旋转向量,外参数
    # print("tvecs:\n", tvecs )   # 平移向量,外参数
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u,v), 0, (u,v))
    # print("newcameramtx", newcameramtx) # 外参数
    return mtx, dist, rvecs, tvecs

# Online phase: Capture picture using webcam
def online(mtx, dist):
    camera = cv2.VideoCapture(0)
    n = 0
    w = 9
    h = 6
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0], [0,0,-2], [0,2,-2],[2,2,-2], [2,0,-2]])

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
             # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            draw(frame, corners2, imgpts)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (w, h), corners2, ret)
            # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('findCorners', 640, 480)
       
        cv2.imshow('Camera', frame)
        cv2.waitKey(200)

        # u, v = frame.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u,v), 0, (u,v))
        # undistort
        # dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # remapping, same result as undistort
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w1,h1), 5)
        # dst2 = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        # x, y, w1, h1 = roi
        # dst1 = dst1[y:y+h1, x:x+w1]
        # cv2.imshow("NewImage", dst1)
        # k = cv2.waitKey(1)
        # if k == 27:   #press Esc to quit
        #     break
        # elif k == ord('s'):   #press s to save the image
        #     cv2.imwrite('./UndistortImage/' + str(n) + '.jpg', dst1)
        #     n += 1

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mtx, dist, rvecs, tvecs = offline()
    online(mtx, dist)