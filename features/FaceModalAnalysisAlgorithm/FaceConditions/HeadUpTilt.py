from features.FaceModalAnalysisAlgorithm.NoseModalities import *


def HeadUpTilt(landmarks, NB_neutral_data):
    return LongitudinalRotation(landmarks, NB_neutral_data)
