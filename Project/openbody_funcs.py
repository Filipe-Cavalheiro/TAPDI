import cv2 as cv
from gameManager import GameManager

def detect_face(frame, net, conf_threshold, verbose=False):   
    """
    Description: Current implementation of face detection uses Haar like features / Viola Jones 
    """
    assert frame is not None, "file could not be read, check with os.path.exists()"
    """ face_rects = haar.detectMultiScale(
        frame,
        scaleFactor = 1.4,
        minSize = (20, 20),
        maxSize = (100,100))
    
    if verbose:
        for rect in face_rects:
            cv.rectangle(frame, rect, (255, 0, 0), 2)
            
    return face_rects """

    frameHeight, frameWidth = frame.shape[:2]
    
    # prepare for DNN
    blob = cv.dnn.blobFromImage(frame,
        scalefactor=1.0,
        size=(800, 800),
        mean= [0, 0, 0],
        swapRB=False,
        crop=False)
    
    #set image as DNN input
    net.setInput(blob)

    #get Output
    detections = net.forward()

    face_rects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            face_rects.append([x1, y1, x2, y2])
            
            
    if verbose:
        for rect in face_rects:
            cv.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 8)
            
    return face_rects

def get_body_roi(frame, face, verbose=False):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face

    fw = x2 - x1
    fh = y2 - y1

    face_center_x = x1 + fw // 2

    # Tunable parameters
    x_margin = int(6 * fw)     # sides
    body_height = int(5.0 * fh)  # downwards

    x1 = max(0, face_center_x - x_margin)
    x2 = min(w, face_center_x + x_margin)
    y1 = max(0, y1) - 40
    y2 = h

    if verbose:
        cv.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return x1, y1, x2, y2

def pose_estimation(frame, gm: GameManager, threshold, verbose=False):
    """
    Runs pose estimation on a resized frame and returns keypoints
    in ORIGINAL image coordinates.
    """

    frameHeight, frameWidth, _ = frame.shape

    # Create blob
    blob = cv.dnn.blobFromImage(
        frame,
        1.0 / 255,
        (frameWidth, frameHeight),
        (0, 0, 0),
        swapRB=False,
        crop=False
    )

    gm.net.setInput(blob)
    out = gm.net.forward()

    H = out.shape[2]
    W = out.shape[3]
   
    # Empty list to store the detected keypoints
    points = []
    for i in range(out.shape[1]):
        # confidence map of corresponding body's part.
        probMap = out[0, i, :, :]
    
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv.minMaxLoc(probMap)
    
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
    
        if prob > threshold :
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    if verbose: 
        for i in range(len(points)):
            cv.circle(frame, points[i], 4, (0, 0, 255), -1)
        """for partFrom, partTo in gm.POSE_PAIRS:
            idFrom = gm.BODY_PARTS[partFrom]
            idTo = gm.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 255), 2)
                cv.circle(frame, points[idFrom], 4, (0, 0, 255), -1)
                cv.circle(frame, points[idTo], 4, (0, 0, 255), -1)"""

    return points
