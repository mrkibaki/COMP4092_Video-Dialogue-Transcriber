import cv2
import dlib
import os
import time
from features.NeutralThresholdDataCollection.InnerEEBDist import *
from features.NeutralThresholdDataCollection.AvgDataCollection import *
from features.NeutralThresholdDataCollection.InnerEBDist import *


def neutral_face_data_collection(video_source=0):
    # Initialize the camera
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Camera resource isn't available.")
        return

    # Load the facial detector and landmark predictor
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "Models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    collection_duration = 5  # Total duration for data collection
    interval = collection_duration / 10  # Interval between collections
    collect_data = False
    EEB_distances_right = []
    EEB_distances_left = []
    EB_dist = []

    try:
        start_time = None  # Start time of collection
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera. Exiting...")
                break

            if collect_data:
                if time.time() - start_time < collection_duration:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    for face in faces:
                        landmarks = predictor(gray, face)

                        # Function call for collecting the neutral distance between eye and eyebrow
                        norm_EEB_dist = InnerEyeNEyeBrowDist(landmarks)
                        EEB_distances_left.append(norm_EEB_dist['left'])
                        EEB_distances_right.append(norm_EEB_dist['right'])
                        norm_EB_dist = InnerEBDist(landmarks)
                        EB_dist.append(norm_EB_dist['ebdist'])
                else:
                    # Once the collection duration is over, break the loop
                    break
            else:
                # Display the message to start data collection
                # 请按下空格开始收集数据，收集数据时间为5秒，请保持自然状态不动
                message = "Press Space to collect data in 5 sec. Please stay neutral still after spacing."
                textsize = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                textX = (frame.shape[1] - textsize[0]) // 2
                textY = (frame.shape[0] + textsize[1]) // 2
                cv2.putText(frame, message, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                if not collect_data:
                    collect_data = True
                    start_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Calculate the average of the collected distances
    AvgDataSet = AvgData(EEB_distances_right, EEB_distances_left, EB_dist)
    print(f"Average Normalized Distance Right: {AvgDataSet['right']:.3f}")
    print(f"Average Normalized Distance Left: {AvgDataSet['left']:.3f}")

    # AvgDataSet need to add EB_dist simutaneously
    print(f"Average Normalized Eyevrow Distance: {AvgDataSet['ebdist']:.3f}")

    return AvgDataSet
