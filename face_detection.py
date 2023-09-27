import cv2
import numpy as np


def detect_face_mark(model_path):
    # initialise face detector and face mark detector
    face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")
    facemark = cv2.face.createFacemarkLBF()
    # eye_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_eye.xml")
    facemark.loadModel(model_path)

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
