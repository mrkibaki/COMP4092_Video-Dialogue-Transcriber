from features.FaceModalAnalysisAlgorithm.BrowModalities import IsFrowning


def FrownCon(landmarks, face, neutral_data):
    if IsFrowning(landmarks, face, neutral_data):
        return True
