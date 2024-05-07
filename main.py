from features.FD_with_dlib import face_detection
from features.neutral_face_data import neutral_face_data_collection


# LFB model math
model_path = "./LBF/lbfmodel.yaml"

# Function calls
neutral_data = neutral_face_data_collection()
face_detection(neutral_data)
