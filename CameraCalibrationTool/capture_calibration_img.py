import cv2
import os


def capture_calibration_images(save_path='CalibrationImages/', chessboard_size=(6, 9)):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)
    img_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if img_counter >= 25:
            break

        cv2.imshow('Calibration', frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = os.path.join(save_path, f"calib_{img_counter}.png")
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cap.release()
    cv2.destroyAllWindows()


# 下面的代码是用于直接运行此脚本时的测试代码
if __name__ == "__main__":
    capture_calibration_images()
