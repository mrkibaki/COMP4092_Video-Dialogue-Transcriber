import cv2
import numpy as np
import glob

# 标定图像的路径
images_path = glob.glob('CalibrationImages/*.png')

# 棋盘格的设置
chessboard_size = (5, 8
                   )
square_size = 1.0

# 准备对象点
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  # 3D 点
imgpoints = []  # 2D 点
image_shape = None  # 用于存储图像的大小

for img_path in images_path:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Detected Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard was not detected in {img_path}")
cv2.destroyAllWindows()

# 确保我们有有效的图像大小
if image_shape is None:
    raise ValueError("没有图像被处理，无法获取图像大小。")

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

print("Camera matrix:", mtx)
print("Distortion coefficients:", dist)
