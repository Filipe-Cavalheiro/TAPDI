import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def main():
    # ex1
    # All the 6 methods for comparison in a list
    methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
                'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
 

    template = cv.imread('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\car template (1).bmp')
    assert template is not None, "file could not be read, check with os.path.exists()"
    # w, h = template.shape[::-1]
    
    # Assign directory
    directory = r'C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\pedestrians'

    # Iterate over files in directory
    for name in os.listdir(directory):
        if name.split('.')[-1] == "mp4":
            cap = cv.VideoCapture('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\pedestrians\\' + str(name))
 
            while cap.isOpened():
                ret, frame = cap.read()
            
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            
                hog = cv.HOGDescriptor()
                hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector() )
                pedestrians = hog.detectMultiScale(frame)
                for (x,y,w,h) in pedestrians[0]:
                    cv.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 4)
            
                cv.imshow('frame', frame)
                if cv.waitKey(1) == ord('q'):
                    break
        else:
            original_img = cv.imread('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\pedestrians\\' + str(name))
            assert original_img is not None, "file could not be read, check with os.path.exists()"
            original_img = cv.cvtColor(original_img, cv.COLOR_RGB2BGR)
            
            hog = cv.HOGDescriptor()
            hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector() )
            pedestrians = hog.detectMultiScale(original_img)
            for (x,y,w,h) in pedestrians[0]:
                cv.rectangle(original_img, (x,y), (x+w,y+h), (0, 0, 255), 4)

            plt.imshow(original_img)
            plt.show()


if __name__ == "__main__":
    main()