import cv2
import numpy as np


def estimate_head_pose(image_points, model_points, camera_matrix, dist_coeffs):
    # 执行头部姿态估计
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

    # 定义一个点，这个点在头部坐标系中位于鼻尖正前方
    nose_end_point3D = np.array([(0.0, 0.0, 1000.0)])

    # 通过旋转和平移向量将这个点转换到相机坐标系中
    nose_end_point2D, jacobian = cv2.projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix,
                                                   dist_coeffs)

    # 计算绘制线条的点
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2
