import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def main():
    # assing cacade model
    detectorFilename = "haarcascade_frontalface_default.xml"
    haar = cv.CascadeClassifier("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\models\\" + detectorFilename)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        face_rects = haar.detectMultiScale(
        frame,
        scaleFactor = 1.4,
        minSize = (20, 20),
        maxSize = (100,100) )


        for rect in face_rects:
            cv.rectangle(frame, rect, (255, 0, 0), 2)

            for (x, y, w, h) in face_rects:
                # Extract the face region of interest (ROI)
                face_roi = frame[y:y+h, x:x+w]

                # Blur the face ROI
                face_blur = cv.GaussianBlur(face_roi, (35, 35), 30)

                # Replace the original face region with the blurred one
                frame[y:y+h, x:x+w] = face_blur

                # Optionally draw a rectangle (for debugging)
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Show the frame with blurred faces
            cv.imshow('Blurred Faces', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()