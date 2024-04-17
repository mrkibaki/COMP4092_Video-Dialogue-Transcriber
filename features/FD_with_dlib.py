import cv2
import dlib
import os
from features.GazeTracking.gaze_tracking.gaze_tracking import GazeTracking
from features.head_pose_estimation import estimate_head_pose

from features.FaceModalAnalysisAlgorithm.MouthModalities import *
from features.FaceModalAnalysisAlgorithm.FrowModalities import *
from features.FaceModalAnalysisAlgorithm.FaceConditions.IsSmiling import IsSmiling


def face_detection():
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "Models/shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()  # 使用HOG检测器
    predictor = dlib.shape_predictor(model_path)

    gaze = GazeTracking()

    cap = cv2.VideoCapture(0)

    # camera calication data from ocv
    camera_matrix = np.array([[832.38183417, 0.0, 686.32961237],
                              [0.0, 838.8275083, 440.29031722],
                              [0.0, 0.0, 1.0]])

    dist_coeffs = np.array([-0.1321268, 0.04720366, 0.02581908, 0.02791332, 0.04725764])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 可以选择进一步缩小图像尺寸来提高速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)  # 使用HOG检测器

        # 这里使用的坐标应根据您的模型参考调整
        model_points = np.array([
            (0.0, 0.0, 0.0),  # 鼻尖
            (0.0, -330.0, -65.0),  # 下巴
            (-225.0, 170.0, -135.0),  # 左眼左角
            (225.0, 170.0, -135.0),  # 右眼右角
            (-150.0, -150.0, -125.0),  # 左嘴角
            (150.0, -150.0, -125.0)  # 右嘴角
        ])

        # 摄像头内参，这里使用的是示例值，应根据您的相机进行调整
        size = frame.shape
        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)

        for face in faces:
            x1 = int(face.left() * 2)
            y1 = int(face.top() * 2)
            x2 = int(face.right() * 2)
            y2 = int(face.bottom() * 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            for n in range(0, 68):
                # mark down all 68 points
                x = int(landmarks.part(n).x * 2)
                y = int(landmarks.part(n).y * 2)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # 模态检视部分，用于查看是否有张嘴或微笑
            if MouthOpening(landmarks):
                cv2.putText(frame, "Mouth is open", (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Mouth is closed", (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 计算是否咧嘴
            Grinning = IsGrinning(landmarks)

            # 显示结果
            lips_status = "Grinning" if Grinning else "Not Grinning"
            cv2.putText(frame, lips_status, (x1, y2 + 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 计算是否微笑
            wlratio = WidthLengthRatio(landmarks)
            Smiling = IsSmiling(Grinning, wlratio)

            # 显示结果
            lips_status = "Smiling" if Smiling else "Not Smiling"
            cv2.putText(frame, lips_status, (x1, y2 + 60), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 显示结果
            lips_status = "Ratio Active" if wlratio else "Ratio not Active"
            cv2.putText(frame, lips_status, (x1, y2 + 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            Frowing = IsFrowing(landmarks)
            # 绘制特定的面部特征点
            for n in [21, 22, 39, 42]:  # 眉毛内侧点和对应的眼角点
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x*2, y*2), 2, (0, 255, 0), -1)

            # 显示结果
            lips_status = "Frowing" if Frowing else "Not Frowing"
            cv2.putText(frame, lips_status, (x1, y2 + 100), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 获取用于 solvePnP 的 2D 点
            image_points = np.array([
                (landmarks.part(33).x * 2, landmarks.part(33).y * 2),  # 鼻尖
                (landmarks.part(8).x * 2, landmarks.part(8).y * 2),  # 下巴
                (landmarks.part(36).x * 2, landmarks.part(36).y * 2),  # 左眼左角
                (landmarks.part(45).x * 2, landmarks.part(45).y * 2),  # 右眼右角
                (landmarks.part(48).x * 2, landmarks.part(48).y * 2),  # 左嘴角
                (landmarks.part(54).x * 2, landmarks.part(54).y * 2)  # 右嘴角
            ], dtype="double")

            # Call the function to estimate head pose
            p1, p2 = estimate_head_pose(image_points, model_points, camera_matrix, dist_coeffs)

            # 绘制线条：从鼻尖到计算出的点
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31),
                    1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

##############################################################
# # Eye pupil
# for eye_points in [range(36, 42), range(42, 48)]:
#     eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in eye_points])
#     x, y, w, h = cv2.boundingRect(eye)
#     eye_image = gray[y:y + h, x:x + w]
#
#     # 应用高斯模糊
#     eye_image = cv2.GaussianBlur(eye_image, (7, 7), 0)
#
#     # 使用高斯模糊减少噪声
#     eye_image_blurred = cv2.GaussianBlur(eye_image, (7, 7), 0)
#
#     # 检测圆 - 虹膜
#     circles = cv2.HoughCircles(eye_image_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30,
#                                minRadius=int(h / 4), maxRadius=int(h / 2))
#
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for (cx, cy, r) in circles:
#             # 绘制虹膜边缘
#             cv2.circle(frame, ((cx + x) * 2, (cy + y) * 2), r * 2, (255, 255, 0), 4)
#             # 绘制虹膜中心
#             cv2.circle(frame, ((cx + x) * 2, (cy + y) * 2), 2, (0, 255, 0), -1)
#
#     # 使用自适应阈值
#     _, thresholded_eye = cv2.threshold(eye_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     # 形态学操作
#     kernel = np.ones((2, 2), np.uint8)
#     thresholded_eye = cv2.erode(thresholded_eye, kernel, iterations=2)
#     thresholded_eye = cv2.dilate(thresholded_eye, kernel, iterations=4)
#     thresholded_eye = cv2.erode(thresholded_eye, kernel, iterations=2)
#
#     contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_blob = max(contours, key=cv2.contourArea)
#         M = cv2.moments(largest_blob)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#             pupil_center_global = ((cx + x) * 2, (cy + y) * 2)
#             cv2.circle(frame, pupil_center_global, 3, (0, 0, 255), -1)


# 可能用到的模态分析
# # 获取眼睛中心点
# left_eye_center = np.mean(np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]), axis=0)
# right_eye_center = np.mean(np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]),
#                            axis=0)
#
# # 计算眼睛中心点之间的角度
# eye_dx = right_eye_center[0] - left_eye_center[0]
# eye_dy = right_eye_center[1] - left_eye_center[1]
# eye_angle = np.degrees(np.arctan2(eye_dy, eye_dx))
#
# # 获取嘴巴两侧点
# mouth_left = landmarks.part(48)
# mouth_right = landmarks.part(54)
#
# # 计算嘴巴两侧点之间的角度
# mouth_dx = mouth_right.x - mouth_left.x
# mouth_dy = mouth_right.y - mouth_left.y
# mouth_angle = np.degrees(np.arctan2(mouth_dy, mouth_dx))
# # 根据眼睛和嘴巴的角度判断面部是否正对摄像头
# if abs(eye_angle) < 10 and abs(mouth_angle) < 10:  # 角度阈值可以根据需要调整
#     print("Face is frontal")
# else:
#     print("Face is not frontal")
#
# ##############################################################
