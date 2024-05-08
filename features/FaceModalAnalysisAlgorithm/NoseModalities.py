from features.NeutralThresholdDataCollection.StableDataOuterEyePoint import *


def NoseFlaring():
    return 0


def LatitudinalRotation():
    return 0


def LongitudinalRotation(landmarks, NB_neutral_data):
    eye_width = OuterEyePointDistance(landmarks)

    # no.27-30 are landmarks for nasal bridge
    p27 = np.array([landmarks.part(27).x, landmarks.part(27).y])
    p28 = np.array([landmarks.part(28).x, landmarks.part(28).y])
    p29 = np.array([landmarks.part(29).x, landmarks.part(29).y])
    p30 = np.array([landmarks.part(30).x, landmarks.part(30).y])

    distance2728 = np.linalg.norm(p27 - p28)
    distance2829 = np.linalg.norm(p28 - p29)
    distance2930 = np.linalg.norm(p29 - p30)
    nadal_bridge_len = np.linalg.norm(p27 - p30)

    # normalise the data
    norm_d1 = distance2728/eye_width
    norm_d2 = distance2829/eye_width
    norm_d3 = distance2930/eye_width
    norm_NB_len = nadal_bridge_len/eye_width

    threshold1 = NB_neutral_data['seg1'] * 0.9
    threshold2 = NB_neutral_data['seg2'] * 0.9
    threshold3 = NB_neutral_data['seg3'] * 0.9
    threshold4 = NB_neutral_data['NB'] * 0.9

    # possibility check
    if norm_d1 < threshold1:
        # 仰头可能性基于长度减小的比例
        possibility1 = (min(1, max(0, np.exp(7 * (1 - norm_d1 / threshold1)) - 1))) * 100
    else:
        possibility1 = 0

    if norm_d2 < threshold2:
        possibility2 = (min(1, max(0, np.exp(7 * (1 - norm_d2 / threshold2)) - 1))) * 100
    else:
        possibility2 = 0

    if norm_d3 < threshold3:
        possibility3 = (min(1, max(0, np.exp(7 * (1 - norm_d3 / threshold3)) - 1))) * 100
    else:
        possibility3 = 0

    if norm_NB_len < threshold4:
        possibility4 = (min(1, max(0, np.exp(3 * (1 - norm_NB_len / threshold4)) - 1))) * 100
    else:
        possibility4 = 0

    nasal_bridge = {'seg1': possibility1, 'seg2': possibility2, 'seg3': possibility3, 'NB': possibility4}

    return nasal_bridge
