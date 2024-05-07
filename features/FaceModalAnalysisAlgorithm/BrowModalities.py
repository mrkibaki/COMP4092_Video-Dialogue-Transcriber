import numpy as np


def IsFrowning(landmarks, face, neutral_data):
    # 获取脸部边界
    right_eye_outer = np.array([landmarks.part(45).x, landmarks.part(45).y])
    left_eye_outer = np.array([landmarks.part(42).x, landmarks.part(42).y])
    eye_width = np.linalg.norm(right_eye_outer - left_eye_outer)

    # 获取眉毛内侧和眼角的点
    right_brow_inner = np.array([landmarks.part(21).x, landmarks.part(21).y])
    left_brow_inner = np.array([landmarks.part(22).x, landmarks.part(22).y])
    right_eye_inner = np.array([landmarks.part(39).x, landmarks.part(39).y])
    left_eye_inner = np.array([landmarks.part(42).x, landmarks.part(42).y])

    # 计算实际距离并归一化
    distance_right = np.linalg.norm(right_brow_inner - right_eye_inner) / eye_width
    distance_left = np.linalg.norm(left_brow_inner - left_eye_inner) / eye_width

    # 打印归一化的距离
    print(f"Normalized Distance Right: {distance_right}")
    print(f"Normalized Distance Left: {distance_left}")

    # 定义阈值
    left_threshold = neutral_data['left']
    right_threshold = neutral_data['right']

    # 判断是否在皱眉
    return distance_right < right_threshold or distance_left < left_threshold
