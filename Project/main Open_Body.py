import cv2 as cv
import numpy as np
from cleanup import cleanup
from openbody_funcs import detect_face
from pose_estimation import pose_estimation
from gameManager import GameManager

def main():
    gm = GameManager()
    
    while True:
        ret, frame = gm.cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        face_rects = detect_face(frame, gm.haar, False)
        
        if len(face_rects) == 0:
            #since nothing was detected show frame prematurely
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                cleanup(gm)
                break
            continue

        x1, y1, x2, y2 = get_body_roi(frame, face_rects[0], False)
        frame = frame[y1:y2, x1:x2]

        frame = down_sampling(frame)
        print(frame.shape)

        points = pose_estimation(frame, gm, gm.THRESHOLD, True)

        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cleanup(gm)
            break

if __name__ == "__main__":
    main()
