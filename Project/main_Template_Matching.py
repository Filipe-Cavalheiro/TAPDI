import cv2 as cv
import numpy as np

from cleanup import cleanup
from gameManager import GameManager
from kernel_funcs import subtract_template_avg, normalized_correlation_coefficient, linear_transform_img

def main():
    gm = GameManager()
    template_minus_avg, template_sum_mean = subtract_template_avg(gm, False)
    
    ret, frame = gm.cap.read()
    frame_h, frame_w, _ =  frame.shape
    temp_h, temp_w, _ = template_minus_avg.shape
    
    while True:
        ret, frame = gm.cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        #correlati
        # on_coefficient(frame, template_minus_avg, template_sum_mean, gm, True)
        output_buff = normalized_correlation_coefficient(frame, template_minus_avg, template_sum_mean, gm, False)
        
        max_index = np.unravel_index(output_buff.argmax(), output_buff.shape)
        
        center_x = max_index[1] + temp_w // 2
        center_y = max_index[0] + temp_h // 2

        dx = frame_w // 2 - center_x
        dy = frame_h // 2 - center_y

        frame = linear_transform_img(gm, frame, np.array([dx,dy]), False)

        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cleanup(gm)
            break
        
if __name__ == "__main__":
    main()
