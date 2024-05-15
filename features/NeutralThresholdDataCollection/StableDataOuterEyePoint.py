import numpy as np
import csv
import os


def OuterEyePointDistance(landmarks):
    # Calculate the Euclidean distance between the outer corners of the eyes
    right_eye_outer = np.array([landmarks.part(45).x, landmarks.part(45).y])
    left_eye_outer = np.array([landmarks.part(36).x, landmarks.part(36).y])
    eye_width = np.linalg.norm(right_eye_outer - left_eye_outer)

    return eye_width
