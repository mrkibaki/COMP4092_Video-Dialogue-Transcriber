import cv2
import numpy as np


def get_eye_region(img, landmarks, points):
    mask = np.zeros_like(img, dtype=np.uint8)
    points = np.array(points, np.int32)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    eye = cv2.bitwise_and(img, mask)
    start_x = points[:, 0].min()
    start_y = points[:, 1].min()
    end_x = points[:, 0].max()
    end_y = points[:, 1].max()

    return eye, (start_x, start_y, end_x, end_y)


def detect_pupils(eye_region, frame, start_point):
    # 转换为灰度图像
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # 使用Otsu's方法进行二值化（threshholding）
    block_size = 11  # 通常为奇数，表示计算阈值时使用的区域大小
    C = 2  # 一个常数，从计算的平均值或加权平均值中减去的常数
    thresh = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)

    # 1. 使用膨胀操作处理阈值化后的图像
    kernel = np.ones((3, 3), np.uint8)
    dilated_eye = cv2.dilate(thresh, kernel, iterations=2)

    # 在Otsu's阈值化后，显示二值化的眼睛图像
    # cv2.imshow("Thresholded Eye", thresh)

    # 查找轮廓
    contours, _ = cv2.findContours(dilated_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # 检查是否有轮廓被检测到
        c = max(contours, key=cv2.contourArea)  # 获取最大的轮廓
        M = cv2.moments(c)

        if M["m00"] != 0:  # 确保分母不为0
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 打印计算的中心坐标
            print("Eye Start Point:", start_point)
            print("Pupil Center:", cX, cY)
            local_cX = int(M["m10"] / M["m00"])
            local_cY = int(M["m01"] / M["m00"])
            print("Local Pupil Center:", local_cX, local_cY)
            # 在原始帧上标记瞳孔中心为红色
            cv2.circle(frame, (cX, cY), 2, (0, 0, 255), -1)
            cv2.drawContours(eye_region, contours, -1, (0, 255, 0), 1)
            cv2.imshow('Contours', eye_region)

        else:
            cX, cY = 0, 0
        return cX, cY

    else:
        return None, None

