import cv2 as cv

def detect_face(frame, haar, verbose=False):   
    # Description:
    # Current implementation of face detection uses Haar like features / Viola Jones 

    assert frame is not None, "file could not be read, check with os.path.exists()"
    face_rects = haar.detectMultiScale(
        frame,
        scaleFactor = 1.4,
        minSize = (20, 20),
        maxSize = (100,100))
    
    if verbose:
        for rect in face_rects:
            cv.rectangle(frame, rect, (255, 0, 0), 2)
            
    return face_rects