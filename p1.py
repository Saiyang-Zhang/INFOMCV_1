import cv2
import numpy as np
import glob
import math
import time

# import win32api,win32con

w = 9
h = 6
# termination criteria, maximum number of loops = 30 and maximum error tolerance = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# checkerboard points in the world coordinate system, e.g. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0),
# with the z-coordinate removed, recorded as a two-dimensional matrix
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * 23    #grid length = 23mm

# Store the world coordinates and image coordinates of the checkerboard grid
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

coordinates = []
subCoord = []
i = 0

# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
    global coordinates, subCoord, i

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # displaying the coordinates on the Shell
        if i < 4:
            cv2.putText(params, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
            cv2.imshow('findCorners', params)
            coordinates.append([x, y])
        i += 1
        if i == 4:
            subcoordinates(params)
            #subcoordinates_alter(params)
            addCorners()

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # displaying the coordinates on the Shell
        if i < 4:
            b = params[y, x, 0]
            g = params[y, x, 1]
            r = params[y, x, 2]
            cv2.putText(params, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
            cv2.imshow('findCorners', params)
            cv2.circle(params, (x, y), 5, (0, 255, 0), -1)
            coordinates.append([x, y])
        i += 1
        if i == 4:
            # subcoordinates(params)
            subcoordinates_alter(params)
            addCorners()

def addCorners():
    global coordinates, subCoord, i
    #win32api.MessageBox(0, "Please close the window", "Warning", win32con.MB_OK)

    # make it the same format as pattern detection results
    corners2 = []
    for Coor in subCoord:
        corners2.append([Coor])

    objpoints.append(objp)
    imgpoints.append(np.array(corners2, np.float32))
    #clear the numbers
    coordinates = []
    subCoord = []
    i = 0

# find the coordinates of each square
def subcoordinates(img):
    SortCoord = sorted(coordinates, key=(lambda x: x[0]))
    # square root ((x2-x1)^2 - (y2-y1)^2)
    len1 = math.sqrt(((SortCoord[1][0] - SortCoord[0][0]) ** 2) + ((SortCoord[1][1] - SortCoord[0][1]) ** 2))
    len2 = math.sqrt(((SortCoord[2][0] - SortCoord[0][0]) ** 2) + ((SortCoord[2][1] - SortCoord[0][1]) ** 2))
    # print(len1, len2)
    if len1 > len2:
        coordLongLeg1 = split(SortCoord[0], SortCoord[1], 8)
        coordLongLeg2 = split(SortCoord[2], SortCoord[3], 8)
    else:
        coordLongLeg1 = split(SortCoord[0], SortCoord[2], 8)
        coordLongLeg2 = split(SortCoord[1], SortCoord[3], 8)

    for m in range(9):
        subSubCoord = split(coordLongLeg1[m], coordLongLeg2[m], 5)
        for n in subSubCoord:
            subCoord.append(n)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = np.array(subCoord, np.float32)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(img, (w, h), corners2, True)
    cv2.imshow('findCorners', img)

# split the line sigment and get the coordinates
def split(start, end, segments):
    x_delta = (end[0] - start[0]) / float(segments)
    y_delta = (end[1] - start[1]) / float(segments)
    points = []
    for i in range(1, segments):
        points.append([start[0] + i * x_delta, start[1] + i * y_delta])
    return [start] + points + [end]

def subcoordinates_alter(img):
    SortCoord = sorted(coordinates, key=(lambda x: x[0]))
    # square root ((x2-x1)^2 - (y2-y1)^2)
    len1 = math.sqrt(((SortCoord[1][0] - SortCoord[0][0]) ** 2) + ((SortCoord[1][1] - SortCoord[0][1]) ** 2))
    len2 = math.sqrt(((SortCoord[2][0] - SortCoord[0][0]) ** 2) + ((SortCoord[2][1] - SortCoord[0][1]) ** 2))
    # print(len1, len2)
    if len1 > len2:
        l1 = []
        interp_xl1 = np.linspace(SortCoord[0][0], SortCoord[1][0], w)
        interp_yl1 = np.linspace(SortCoord[0][1], SortCoord[1][1], w)
        for n in range(w):
            l1.append([interp_xl1[n], interp_yl1[n]])
        l2 = []
        interp_xl2 = np.linspace(SortCoord[2][0], SortCoord[3][0], w)
        interp_yl2 = np.linspace(SortCoord[2][1], SortCoord[3][1], w)
        for n in range(w):
            l2.append([interp_xl2[n], interp_yl2[n]])
    else:
        l1 = []
        interp_xl1 = np.linspace(SortCoord[0][0], SortCoord[2][0], w)
        interp_yl1 = np.linspace(SortCoord[0][1], SortCoord[2][1], w)
        for n in range(w):
            l1.append([interp_xl1[n], interp_yl1[n]])
        l2 = []
        interp_xl2 = np.linspace(SortCoord[1][0], SortCoord[3][0], w)
        interp_yl2 = np.linspace(SortCoord[1][1], SortCoord[3][1], w)
        for n in range(w):
            l2.append([interp_xl2[n], interp_yl2[n]])

    for n in range(w):
        interp_xs = np.linspace(l1[n][0], l2[n][0], h)
        interp_ys = np.linspace(l1[n][1], l2[n][1], h)
        for m in range(h):
            subCoord.append([interp_xs[m], interp_ys[m]])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = np.array(subCoord, np.float32)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(img, (w, h), corners2, True)
    cv2.imshow('findCorners', img)



# first run
def firstRun():
    images = glob.glob('./CameraRoll/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        # h1, w1 = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

        # If found, save the information
        if ret == True:
            # Finding sub-pixel corners based on the original corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (w, h), corners2, ret)
            # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('findCorners', 640, 480)
            # cv2.imshow('findCorners', img)
            # cv2.waitKey(200)
        else:
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.setMouseCallback('findCorners', click_event, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

# second and third run
def run(round):
    images = glob.glob('./CameraRoll/*.jpg')
    i = 0
    for fname in images:
        if i < round:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            u, v = img.shape[:2]
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
            # If found, save the information
            if ret == True:
                # Finding sub-pixel corners based on the original corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (w, h), corners2, ret)
                # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('findCorners', 640, 480)
                # cv2.imshow('findCorners', img)
                # cv2.waitKey(2000)
                i += 1
        else:
            break
        cv2.destroyAllWindows()
    return gray, u, v

# offline phase
def offline():
    firstRun()  # run1
    run(10)     # run2
    gray, u, v = run(5)     # run3
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Save camera parameters to an NPY file
    np.save('./CameraParams/mtx.npy', mtx)
    np.save('./CameraParams/dist.npy', dist)
    np.save('./CameraParams/rvecs.npy', rvecs)
    np.save('./CameraParams/tvecs.npy', tvecs)

# draw contours online
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -2)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 2)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)
    return img

def shadow(img, shadowpts):
    
    return 

# Online phase: Capture picture using webcam
def online(mtx, dist, rvecs, tvecs):
    camera = cv2.VideoCapture(0)

    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
           
    light = np.float32([[9, 4, -6]])
    #light = light.reshape((1, 1, 2))

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            
            t = (time.time()%12)/6
            x_var = np.cos(t*np.pi)
            z_var = np.sin(t*np.pi)
            axis = np.float32([[x_var, 0, z_var], [x_var, 2, z_var], [-x_var, 2, z_var], [-x_var, 0, z_var], [x_var, 0, -z_var], [x_var, 2, -z_var], [-x_var, 2, -z_var], [-x_var, 0, -z_var]]) 
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            draw(frame, imgpts)

            shadowpts = np

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (w, h), corners2, ret)

        cv2.imshow('Camera', frame)
        k = cv2.waitKey(1)
        if k == 27 or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:   #press Esc to quit
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #offline()
    mtx = np.load('./CameraParams/mtx.npy')
    dist = np.load('./CameraParams/dist.npy')
    rvecs = np.load('./CameraParams/rvecs.npy')
    tvecs = np.load('./CameraParams/tvecs.npy')
    online(mtx, dist, rvecs, tvecs)

