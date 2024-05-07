from features.NeutralThresholdDataCollection.StableDataOuterEyePoint import *


def InnerEBDist(landmarks):
    eye_width = OuterEyePointDistance(landmarks)

    left_brow_inner = np.array([landmarks.part(22).x, landmarks.part(22).y])
    right_brow_inner = np.array([landmarks.part(21).x, landmarks.part(21).y])

    norm_brow_dist = np.linalg.norm(right_brow_inner - left_brow_inner)/eye_width
    print(f"norm eyebrow dist is: {norm_brow_dist}")

    distance = {'ebdist': norm_brow_dist}

    return distance
