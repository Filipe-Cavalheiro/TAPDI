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
from kernel_funcs import subtract_template_avg, normalized_correlation_coefficient, linear_transform_img

def main():
    pose = PoseEstimator()
    template_minus_avg, template_sum_mean = subtract_template_avg(pose, False)
    
    ret, frame = pose.cap.read()
    frame_h, frame_w, _ =  frame.shape
    temp_h, temp_w, _ = template_minus_avg.shape
    
    while True:
        ret, frame = pose.cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        #correlati
        # on_coefficient(frame, template_minus_avg, template_sum_mean, pose, True)
        output_buff = normalized_correlation_coefficient(frame, template_minus_avg, template_sum_mean, pose, False)
        
        max_index = np.unravel_index(output_buff.argmax(), output_buff.shape)
        
        center_x = max_index[1] + temp_w // 2
        center_y = max_index[0] + temp_h // 2

        dx = frame_w // 2 - center_x
        dy = frame_h // 2 - center_y

        frame = linear_transform_img(pose, frame, np.array([dx,dy]), False)

        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cleanup(pose)
            break
        
if __name__ == "__main__":
    main()
