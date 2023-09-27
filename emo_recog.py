import cv2
import numpy as np


def emo_recog(image_path):
    # 载入模型
    net = cv2.dnn.readNetFromCaffe("path.prototxt", "path.caffemodel")

    # 预处理图像
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(image, 1.0, (48, 48), (104.0, 177.0, 123.0))

    # 设置网络输入并执行前向传播
    net.setInput(blob)
    predictions = net.forward()

    # 根据输出确定表情
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    emotion = EMOTIONS[np.argmax(predictions[0])]

    return emotion