import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imageOpticalFlow import LucasKanade_OF, Farneback_OF
import os

def main():
    # Assign directory
    directory = r'.\\TP10\\aula 10-video\\'


    # Iterate over files in directory
    for name in os.listdir(directory):
        LucasKanade_OF(directory + name, 3)
        Farneback_OF(directory + name)
        
        """
        cap = cv.VideoCapture(directory + name)

        while cap.isOpened():
            ret, frame = cap.read()
    
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
            cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break"""


if __name__ == "__main__":
    main()