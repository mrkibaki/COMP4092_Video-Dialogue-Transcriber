# import cv2
from face_detection import detect_face_mark


model_path = "./LBF/lbfmodel.yaml"
detect_face_mark(model_path)





# # getting the cascade from the image
# face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
# facemark = cv2.face.createFacemarkLBF()
# eye_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_eye.xml')
#
# # Video file || camera
# # vid = cv2.VideoCapture('')
# vid = cv2.VideoCapture(0)
# # counting eye looking at camera
# look_count = 0
# # frames that looking at the camera
# consecutive_look = 0q
# # consecutive frames with 5 would be valid looking
# Threshhold = 10
# # prev_look = False
#
# while vid.isOpened():
#     ret, frame = vid.read() #reading frames
#     if not ret:
#         break;
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#     looking = False
#     for (x, y, w, h) in face:
#      # framing the face
#      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#      roi_gray = gray[y:y + h, x:x + w]
#      roi_color = frame[y:y + h, x:x + w]
#      # convert the upper half img to gray
#      roi_gray_upper = roi_gray[0:int(0.5 * h), :]
#      eyes = eye_cascade.detectMultiScale(roi_gray_upper, 1.1, 4)
#      for (ex, ey, ew, eh) in eyes:
#          cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#      # determine eyelooking when 2 eyes showing on the screen
#      if len(eyes) == 2:
#          if abs(eyes[0][1] - eyes[1][1]) < h*0.1:
#              eye_distance = abs(eyes[0][0] - eyes[1][0])
#              if w*0.2 < eye_distance < w*0.7:
#                     looking = True
#
#     if looking:
#         consecutive_look += 1
#     else:
#         consecutive_look = 0
#
#     if consecutive_look == Threshhold:
#         look_count += 1
#
#      # if not prev_look and looking:
#      #     look_count += 1
#
#      # prev_look = looking
#
#
#
#
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# cv2.destroyAllWindows()
# print(f"Looked at the camera {look_count} times.")