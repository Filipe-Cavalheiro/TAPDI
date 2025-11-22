import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def getFaceBox(frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    directory = r'TP10\\aula 10-images_files\\models\\'

    # Open DNN model
    modelFile = directory + "opencv_face_detector_uint8.pb"
    configFile = directory + "opencv_face_detector.pbtxt"
    net = cv.dnn.readNetFromTensorflow(modelFile, configFile)
    # prepare for DNN
    blob = cv.dnn.blobFromImage(frameOpencvDnn,
    scalefactor=1.0,
    size=(800, 800),
    mean= [104, 117, 123],
    swapRB=False,
    crop=False)
    #set image as DNN input
    net.setInput(blob)
    #get Output
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0),
    int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def main():
    # Assign directory
    directory = r'C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP10\\aula9\\faces'

    # Iterate over files in directory
    for name in os.listdir(directory):
        original_img = cv.imread(directory + "\\" + str(name))
        assert original_img is not None, "file could not be read, check with os.path.exists()"
        frameOpencvDnn, box = getFaceBox(original_img)
        frameOpencvDnn = cv.cvtColor(frameOpencvDnn, cv.COLOR_BGR2RGB)

        plt.plot()
        plt.imshow(frameOpencvDnn)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    
        plt.show()


if __name__ == "__main__":
    main()