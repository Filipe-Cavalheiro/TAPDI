import cv2 as cv
import numpy as np
from cleanup import cleanup
from detect_face import detect_face
from get_roi import get_body_roi
from down_sampling import down_sampling
from pose_estimation import pose_estimation
from PoseEstimator import PoseEstimator

def main():
    pose = PoseEstimator()
    
    while True:
        ret, frame = pose.cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        face_rects = detect_face(frame, pose.haar, False)
        
        if len(face_rects) == 0:
            #since nothing was detected show frame prematurely
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                cleanup(pose)
                break
            continue

        x1, y1, x2, y2 = get_body_roi(frame, face_rects[0], False)
        frame = frame[y1:y2, x1:x2]

        frame = down_sampling(frame)
        print(frame.shape)

        points = pose_estimation(frame, pose, pose.THRESHOLD, True)

        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cleanup(pose)
            break

if __name__ == "__main__":
    main()
