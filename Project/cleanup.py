import cv2 as cv

def cleanup(pose):
    if pose != None:
        pose.cap.release()
    cv.destroyAllWindows()