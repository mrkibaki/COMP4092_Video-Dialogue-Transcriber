from features.NeutralThresholdDataCollection.StableDataOuterEyePoint import *


# EEB = Eye and Eyebrow
# EB = Eyebrow
def InnerEEBDist(landmarks, neutral_data):
    # 获取脸部边界
    eye_width = OuterEyePointDistance(landmarks)

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
    left_threshold = neutral_data['left'] * 0.9
    right_threshold = neutral_data['right'] * 0.9
    # 目前的阈值是自然状态下的表情，需要调整到皱眉的阈值（可能是自然状态距离的0.97？问问gpt）

    # 判断是否在皱眉
    return distance_right < right_threshold and distance_left < left_threshold


def InnerEBDist(landmarks, neutral_data):
    left_brow_inner = np.array([landmarks.part(22).x, landmarks.part(22).y])
    right_brow_inner = np.array([landmarks.part(21).x, landmarks.part(21).y])
    brow_dist = np.linalg.norm(right_brow_inner - left_brow_inner)

    eye_width = OuterEyePointDistance(landmarks)
    norm_brow_dist = brow_dist/eye_width
    # 打印归一化的距离
    print(f"Normalized Distance Eyebrow: {norm_brow_dist}")

    threshold = neutral_data['ebdist'] * 0.8

    return norm_brow_dist < threshold
