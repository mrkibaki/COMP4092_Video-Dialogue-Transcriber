from features.FaceModalAnalysisAlgorithm.BrowModalities import IsFrowning


def FrownCon(landmarks, face, lrdata):
    if IsFrowning(landmarks, face, lrdata):
        return True
