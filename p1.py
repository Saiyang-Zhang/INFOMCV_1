import cv2
import numpy as np
import glob
import win32api,win32con
import time

w = 9
h = 6
# termination criteria, maximum number of loops = 30 and maximum error tolerance = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# checkerboard points in the world coordinate system 3D
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

# checkerboard points in the world coordinate system 2D, grid length = 23mm
subWorkCoord = np.zeros((w * h, 2), np.float32)
subWorkCoord[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * 23
subWorkCoord = np.array(subWorkCoord, np.float32)

# Store the world coordinates and image coordinates of the checkerboard grid
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Store the mouse click number, click coordinates, manually computed image coordinates
clickNum = 0
coordinates = []
subCoord = []

# Display the coordinates of the points clicked on the image and find all points
def click_event(event, x, y, flags, params):
    global coordinates, subCoord, clickNum
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # displaying the coordinates on the Shell
        if clickNum < 4:
            cv2.putText(params, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
            cv2.imshow('findCorners', params)
            coordinates.append([x, y])
        clickNum += 1
        if clickNum == 4:
            subcoordinates(params)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        # displaying the coordinates on the Shell
        if clickNum < 4:
            b = params[y, x, 0]
            g = params[y, x, 1]
            r = params[y, x, 2]
            cv2.putText(params, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
            cv2.imshow('findCorners', params)
            cv2.circle(params, (x, y), 5, (0, 255, 0), -1)
            coordinates.append([x, y])
        clickNum += 1
        if clickNum == 4:
            subcoordinates(params)

# Calculate the transformation matrix and use it to find all points
def subcoordinates(img):
    global coordinates, subCoord, clickNum

    # compute all coordinates
    worldCoord = np.array([[0,(h-1)*23], [(w-1)*23,(h-1)*23], [(w-1)*23,0], [0,0]], np.float32)
    coordinates_array = np.array(coordinates, np.float32)
    M = cv2.getPerspectiveTransform(worldCoord, coordinates_array)
    res = cv2.perspectiveTransform(subWorkCoord.reshape(-1, 1, 2), M)

    # show the chessboard grid
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = np.array(res, np.float32)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    cv2.drawChessboardCorners(img, (w, h), corners2, True)
    cv2.imshow('findCorners', img)

    # save the corners
    objpoints.append(objp)
    imgpoints.append(np.array(corners2, np.float32))

    #clear the numbers
    coordinates = []
    subCoord = []
    clickNum = 0

# first run
def firstRun():
    images = glob.glob('./CameraRoll/*.jpg')
    i = 0
    for fname in images:
        img = cv2.imread(fname)
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
            cv2.drawChessboardCorners(img, (w, h), corners2, ret)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.waitKey(2000)
        else:
            if i == 0:
                win32api.MessageBox(0, "If detect corners fail, please choose 4 corners clockwise, "
                                       "starting from the top-left.","Notice", win32con.MB_OK)
                i += 1
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
                cv2.drawChessboardCorners(img, (w, h), corners2, ret)
                cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('findCorners', 640, 480)
                cv2.imshow('findCorners', img)
                cv2.waitKey(2000)
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
    print(len(objpoints), len(imgpoints) )
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
            var1 = np.cos(t*np.pi)*np.sqrt(2)
            var2 = np.sin(t*np.pi)*np.sqrt(2)
            axis = np.float32([[1+var1, 1+var2, 0], [1-var2, 1+var1, 0], [1-var1, 1-var2, 0], [1+var2, 1-var1, 0], [1+var1, 1+var2, -2], [1-var2, 1+var1, -2], [1-var1, 1-var2, -2], [1+var2, 1-var1, -2]])
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

