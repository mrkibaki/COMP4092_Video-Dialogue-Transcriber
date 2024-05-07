import cv2
import dlib
import numpy as np
import os
import time


def neutral_face_data_collection(video_source=0):
    # 初始化摄像头
    cap = cv2.VideoCapture(video_source)

    # 加载面部检测器和标志点预测器
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "Models/shape_predictor_68_face_landmarks.dat")

    detector = dlib.get_frontal_face_detector()  # 使用HOG检测器
    predictor = dlib.shape_predictor(model_path)

    # 初始化计时器
    start_time = time.time()
    collection_duration = 5  # 收集数据的时间长度（秒）

    # 收集数据
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            # 计算脸部宽度和高度，用于归一化
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()

            landmarks = predictor(gray, face)

            # 示例：计算眉毛与眼角的垂直和水平距离
            right_brow_inner = landmarks.part(21)
            right_eye_inner = landmarks.part(39)
            left_brow_inner = landmarks.part(22)
            left_eye_inner = landmarks.part(42)
            vertical_distance = abs(right_brow_inner.y - right_eye_inner.y)
            horizontal_distance = abs(right_brow_inner.x - right_eye_inner.x)

            # 归一化距离
            normalized_vertical = vertical_distance / face_height
            normalized_horizontal = horizontal_distance / face_width

            # 显示结果
            cv2.putText(frame, f'Norm Vert: {normalized_vertical:.2f}, Norm Horiz: {normalized_horizontal:.2f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > collection_duration:
            break

    cap.release()
    cv2.destroyAllWindows()
