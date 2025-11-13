import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def main():    
    # Assign directory
    directory = r'C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\faces'

    "assing cacade model"
    detectorFilename = "haarcascade_frontalface_default.xml"
    haar = cv.CascadeClassifier("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\models\\" + detectorFilename)

    # Iterate over files in directory
    for name in os.listdir(directory):
        print('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\faces\\' + str(name))

        original_img = cv.imread('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\faces\\' + str(name))
        assert original_img is not None, "file could not be read, check with os.path.exists()"
        original_img = cv.cvtColor(original_img, cv.COLOR_RGB2BGR)
        face_rects = haar.detectMultiScale(
        original_img,
        scaleFactor = 1.4,
        minSize = (20, 20),
        maxSize = (100,100) )


        for rect in face_rects:
            cv.rectangle(original_img, rect, (255, 0, 0), 2)

        plt.imshow(original_img)
        plt.show()


if __name__ == "__main__":
    main()