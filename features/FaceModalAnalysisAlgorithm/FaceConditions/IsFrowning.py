from features.FaceModalAnalysisAlgorithm.BrowModalities import IsFrowning


def FrownCon(landmarks):
    if IsFrowning(landmarks):
        return True
