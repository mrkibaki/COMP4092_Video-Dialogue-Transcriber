import cv2
import numpy as np
import os
from features.UnusedComponents.eye_detection import get_eye_region, detect_pupils


def detect_face_mark(model_path):
    # initialise face detector and face mark detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    # eye_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_eye.xml")
    facemark.loadModel(model_path)

    print(os.getcwd())

    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        # 从摄像头读取帧
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测脸部
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces_array = np.asarray(faces)

        # 获取面部关键点
        if len(faces) > 0:
            ok, landmarks = facemark.fit(gray, faces_array)

            if ok:
                for marks in landmarks[0]:
                    # left-eye key points from 36 to 42
                    left_eye_points = marks[36:42]
                    # right-eye key points from 42 to 48
                    right_eye_points = marks[42:48]

                    # blue rect to circle whole eyes
                    left_eye_region, (lx1, ly1, lx2, ly2) = get_eye_region(frame, marks, left_eye_points)
                    right_eye_region, (rx1, ry1, rx2, ry2) = get_eye_region(frame, marks, right_eye_points)
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

                    # red dots in eyes' centre
                    detect_pupils(left_eye_region, frame, (lx1, ly1))
                    detect_pupils(right_eye_region, frame, (rx1, ry1))

                    for mark in marks:
                        # print(mark)
                        # 68 key points stored in mark
                        x, y = mark
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        cv2.imshow("Facial Landmarks", frame)

        # 如果按下'q'键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
