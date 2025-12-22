import cv2 as cv
from PoseEstimator import PoseEstimator

def pose_estimation(frame, pose: PoseEstimator, threshold, verbose=False):
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

    pose.net.setInput(blob)
    out = pose.net.forward()

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
        else :
            points.append(None)

    if verbose: 
        for i in range(len(points)):
            cv.circle(frame, points[i], 4, (0, 0, 255), -1)
        """for partFrom, partTo in pose.POSE_PAIRS:
            idFrom = pose.BODY_PARTS[partFrom]
            idTo = pose.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 255), 2)
                cv.circle(frame, points[idFrom], 4, (0, 0, 255), -1)
                cv.circle(frame, points[idTo], 4, (0, 0, 255), -1)"""

    return points
