import cv2 as cv
import numpy as np
import pyopencl as cl
import math


from cleanup import cleanup
from detect_face import detect_face
from get_roi import get_body_roi
from down_sampling import down_sampling
from pose_estimation import pose_estimation
from PoseEstimator import PoseEstimator
from kernel_funcs import subtract_template_avg, normalized_correlation_coefficient

def main():
    pose = PoseEstimator()
    template_minus_avg, template_sum_mean = subtract_template_avg(pose, False)

    """ 
    split_sub_image = cv.split(template_minus_avg)
    flat_minus_avg = np.array([np.sum(np.square(np.array(x).flatten())) for x in split_sub_image])

    cpu_correlation_coefficient = np.mean(flat_minus_avg[:3])

    print((cpu_correlation_coefficient / (np.sqrt(template_sum_mean * template_sum_mean)) + 1) * 127.5, cpu_correlation_coefficient, template_sum_mean)

    frame = cv.imread(".\\Project\\face_template.png")
    correlation_coefficient_no_attomic(frame, template_minus_avg, template_sum_mean, pose, True)
    #correlation_coefficient(frame, template_minus_avg, pose, True) 
    return """
   
    while True:
        ret, frame = pose.cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        #correlation_coefficient(frame, template_minus_avg, template_sum_mean, pose, True)
        output_buff = normalized_correlation_coefficient(frame, template_minus_avg, template_sum_mean, pose, False)

        cv.imshow("frame", output_buff)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cleanup(pose)
            break
        
if __name__ == "__main__":
    main()
