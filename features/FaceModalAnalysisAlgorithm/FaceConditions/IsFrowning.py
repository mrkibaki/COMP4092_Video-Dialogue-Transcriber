from features.FaceModalAnalysisAlgorithm.BrowModalities import *


def FrownCon(landmarks, neutral_data):
    if InnerEEBDist(landmarks, neutral_data) or InnerEBDist(landmarks, neutral_data):
        return True
