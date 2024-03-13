import cv2
import dlib
import os
import numpy as np
from features.GazeTracking.gaze_tracking.gaze_tracking import GazeTracking

current_path = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_path, "tester.mp4")
model_path = os.path.join(current_path, "shape_predictor_68_face_landmarks.dat")

cnn_model_path = os.path.join(current_path, "mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)

predictor = dlib.shape_predictor(model_path)
gaze = GazeTracking()

cap = cv2.VideoCapture(video_path)

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义VideoWriter对象
output_path = os.path.join(current_path, "output_video.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 可以选择进一步缩小图像尺寸来提高速度
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = cnn_face_detector(gray, 1)  # 使用MTCNN

    for face in faces:
        x1 = int(face.rect.left() * 2)
        y1 = int(face.rect.top() * 2)
        x2 = int(face.rect.right() * 2)
        y2 = int(face.rect.bottom() * 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face.rect)
        for n in range(0, 68):
            x = int(landmarks.part(n).x * 2)
            y = int(landmarks.part(n).y * 2)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        ##############################################################
        # detect the face frontal
        # 获取眼睛中心点
        left_eye_center = np.mean(np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]), axis=0)
        right_eye_center = np.mean(np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]),
                                   axis=0)

        # 计算眼睛中心点之间的角度
        eye_dx = right_eye_center[0] - left_eye_center[0]
        eye_dy = right_eye_center[1] - left_eye_center[1]
        eye_angle = np.degrees(np.arctan2(eye_dy, eye_dx))

        # 获取嘴巴两侧点
        mouth_left = landmarks.part(48)
        mouth_right = landmarks.part(54)

        # 计算嘴巴两侧点之间的角度
        mouth_dx = mouth_right.x - mouth_left.x
        mouth_dy = mouth_right.y - mouth_left.y
        mouth_angle = np.degrees(np.arctan2(mouth_dy, mouth_dx))
        # 根据眼睛和嘴巴的角度判断面部是否正对摄像头
        if abs(eye_angle) < 10 and abs(mouth_angle) < 10:  # 角度阈值可以根据需要调整
            print("Face is frontal")
        else:
            print("Face is not frontal")

    ##############################################################

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
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # 保存帧到输出视频
    out.write(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
out.release()  # 释放VideoWriter对象
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
