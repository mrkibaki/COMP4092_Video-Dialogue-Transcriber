import numpy as np


def IsFrowning(landmarks):
    # 获取眉毛内侧和眼角的点
    right_brow_inner = np.array([landmarks.part(21).x, landmarks.part(21).y])
    left_brow_inner = np.array([landmarks.part(22).x, landmarks.part(22).y])
    right_eye_outer = np.array([landmarks.part(39).x, landmarks.part(39).y])
    left_eye_outer = np.array([landmarks.part(42).x, landmarks.part(42).y])

    # 计算垂直距离
    vertical_distance_right = np.abs(right_brow_inner[1] - right_eye_outer[1])
    vertical_distance_left = np.abs(left_brow_inner[1] - left_eye_outer[1])

    # 计算水平距离以归一化
    horizontal_distance_right = np.abs(right_brow_inner[0] - right_eye_outer[0])
    horizontal_distance_left = np.abs(left_brow_inner[0] - left_eye_outer[0])

    # 归一化垂直距离
    normalized_distance_right = vertical_distance_right / horizontal_distance_right
    normalized_distance_left = vertical_distance_left / horizontal_distance_left

    # 定义阈值
    threshold = 0.6

    # 判断是否在皱眉
    return normalized_distance_right < threshold and normalized_distance_left < threshold
