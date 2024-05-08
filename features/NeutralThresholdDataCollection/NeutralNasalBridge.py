from features.NeutralThresholdDataCollection.StableDataOuterEyePoint import *


def NeutralNasalBridge(landmarks):
    eye_width = OuterEyePointDistance(landmarks)

    # no.27-30 are landmarks for nasal bridge
    p27 = np.array([landmarks.part(27).x, landmarks.part(27).y])
    p28 = np.array([landmarks.part(28).x, landmarks.part(28).y])
    p29 = np.array([landmarks.part(29).x, landmarks.part(29).y])
    p30 = np.array([landmarks.part(30).x, landmarks.part(30).y])

    distance2728 = np.linalg.norm(p27 - p28)
    distance2829 = np.linalg.norm(p28 - p29)
    distance2930 = np.linalg.norm(p29 - p30)
    nasal_bridge_len = np.linalg.norm(p27 - p30)

    # normalise the data
    norm_d1 = distance2728/eye_width
    norm_d2 = distance2829/eye_width
    norm_d3 = distance2930/eye_width
    norm_NB_len = nasal_bridge_len/eye_width
    nasal_bridge = {'seg1': norm_d1, 'seg2': norm_d2, 'seg3': norm_d3, 'NB': norm_NB_len}

    return nasal_bridge


def NeutralNasalSeptum(landmarks):
    return 0
