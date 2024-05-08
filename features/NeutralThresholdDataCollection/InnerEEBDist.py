from features.NeutralThresholdDataCollection.StableDataOuterEyePoint import *


def InnerEyeNEyeBrowDist(landmarks):
    eye_width = OuterEyePointDistance(landmarks)
    # Calculate the actual distance between eyebrow and eye corner
    right_brow_inner = np.array([landmarks.part(21).x, landmarks.part(21).y])
    right_eye_inner = np.array([landmarks.part(39).x, landmarks.part(39).y])
    left_brow_inner = np.array([landmarks.part(22).x, landmarks.part(22).y])
    left_eye_inner = np.array([landmarks.part(42).x, landmarks.part(42).y])

    # Normalize distances
    norm_distance_right = np.linalg.norm(right_brow_inner - right_eye_inner) / eye_width
    norm_distance_left = np.linalg.norm(left_brow_inner - left_eye_inner) / eye_width
    print(f"norm dist left is: {norm_distance_left}")
    print(f"norm dist right is: {norm_distance_right}")

    distance = {'left': norm_distance_left, 'right': norm_distance_right}
    return distance
