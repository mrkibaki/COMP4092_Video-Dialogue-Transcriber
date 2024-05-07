# This MAA is based on lnadmarks from 68 landmarks predictor. The specfic landmark points are 49, 55, 61-68. The
# further ladndmarks might also be utilised to enhance the algorthm to be more accuratly. To avoid further redundancy
# changes on passing landmarks in the components, all landmarks related to mouth will be passing in this component.

# This component is used to detect whether the mouth is close or not depending on whether 61-68 landmarks are in the
# same line and within a thresholds
import numpy as np


def IsGrinning(landmarks):
    # 获取关键面部点坐标
    left_eye_inner = np.array([landmarks.part(39).x, landmarks.part(39).y])
    right_eye_inner = np.array([landmarks.part(42).x, landmarks.part(42).y])
    left_corner = np.array([landmarks.part(48).x, landmarks.part(48).y])
    right_corner = np.array([landmarks.part(54).x, landmarks.part(54).y])

    # 计算嘴角和眼角之间的距离
    mouth_width = np.linalg.norm(right_corner - left_corner)
    eye_distance = np.linalg.norm(right_eye_inner - left_eye_inner)

    # 设置比率阈值，需要根据具体数据调整
    ratio_threshold = 1.5

    # 计算比率并判断
    ratio = mouth_width / eye_distance
    return ratio > ratio_threshold


def InnerLipCon(landmarks, threshold=10):
    # 内嘴唇的点位对
    upper_lip_points = [61, 62, 63]
    lower_lip_points = [67, 66, 65]

    for upper, lower in zip(upper_lip_points, lower_lip_points):
        # 上下对应点位的坐标
        upper_point = np.array([landmarks.part(upper).x, landmarks.part(upper).y])
        lower_point = np.array([landmarks.part(lower).x, landmarks.part(lower).y])

        # 计算两点间的欧式距离
        distance = np.linalg.norm(upper_point - lower_point)

        # 如果任一对点之间的距离大于阈值，则认为嘴唇未重合
        if distance > threshold:
            return True

    # 所有点对距离都小于或等于阈值时，认为嘴唇重合
    return False


# This function is used to calculate the ratio of mouth width to height. The possibility of having an open mouth is
# observed by adjusting the threshold of the ratio.
def WidthLengthRatio(landmarks):
    # 计算嘴部宽度
    mouth_width = np.linalg.norm(
        np.array([landmarks.part(48).x, landmarks.part(48).y]) -
        np.array([landmarks.part(54).x, landmarks.part(54).y])
    )

    # 计算嘴部高度
    mouth_height = np.linalg.norm(
        np.array([landmarks.part(51).x, landmarks.part(51).y]) -
        np.array([landmarks.part(57).x, landmarks.part(57).y])
    )

    # 计算高度与宽度的比例
    ratio = mouth_height / mouth_width

    # 定义张嘴的阈值
    threshold = 0.3

    return ratio > threshold


def MouthOpening(landmarks):
    # return WidthLengthRatio(landmarks)
    if InnerLipCon(landmarks):
        return True


