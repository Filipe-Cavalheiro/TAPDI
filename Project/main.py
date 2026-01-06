import cv2 as cv
import numpy as np
import time

from PoseEstimator import PoseEstimator
from openbody_funcs import detect_face, pose_estimation, get_body_roi
from kernel_funcs import subtract_template_avg, normalized_correlation_coefficient, linear_transform_img
from game_funcs import display_image, init_round, restart_game
from cleanup import cleanup

def main():
    loadding_screen = cv.imread(".\\Project\\loading_screen.png")
    cv.namedWindow("loadding screen", cv.WINDOW_NORMAL)
    cv.setWindowProperty("loadding screen", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)  
    cv.imshow("loadding screen", loadding_screen) 
    pose = PoseEstimator()
    template_minus_avg, template_sum_mean = subtract_template_avg(pose, False)
    scored = 0
    game = 0
    wait = 0
    game_start_time = 0
    time_per_round = pose.time_per_round
    score = f"Score: {scored}"

    color_to_select = None
    text_color = None
    color_text = None

    ret, frame = pose.cap.read()
    frame_h, frame_w, _ =  frame.shape
    temp_h, temp_w, _ = template_minus_avg.shape
    
    top_left_corner = (20, 40)
    
    cv.destroyWindow("loadding screen")
    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN) 

    while True:
        start_time = time.time()
        ret, frame = pose.cap.read()

        frame = cv.flip(frame, 1)

        if not ret:
            print("Can't receive frame. Exiting...")
            break

        output_buff = normalized_correlation_coefficient(frame, template_minus_avg, template_sum_mean, pose, False)
        
        max_index = np.unravel_index(output_buff.argmax(), output_buff.shape)
        
        center_x, center_y =  np.array(output_buff.shape) // 2

        center_x = max_index[1] + temp_w // 2
        center_y = max_index[0] + temp_h // 2

        dx = frame_w // 2 - center_x
        dy = frame_h // 5 - center_y

        face_rects = detect_face(frame, pose.DNN, pose.face_detection_threshold, False)
        if len(face_rects) == 0:
            #since nothing was detected show frame prematurely
            display_image(frame, top_left_corner, start_time, 0, time_per_round, pose)
            if cv.waitKey(1) & 0xFF == ord("q"):
                cleanup(pose)
                break
            continue

        x1, y1, x2, y2 = get_body_roi(frame, face_rects[0], False)
        roi_frame = frame[y1:y2, x1:x2]
        h, w = roi_frame.shape[:2]

        if roi_frame is None:

            display_image(frame, top_left_corner, start_time, 0, time_per_round, pose)
            if cv.waitKey(1) & 0xFF == ord("q"):
                cleanup(pose)
                break
            continue
        
        try:
            roi_frame = cv.resize(roi_frame, (pose.crop_size, pose.crop_size))
        except Exception as e:
            print(type(roi_frame))
            print(roi_frame)
            print(f"roi frame shape: {roi_frame.shape}")
            print(f"{e}")

        points = pose_estimation(roi_frame, pose, pose.THRESHOLD, False)

        # roi_frame = cv.resize(roi_frame, (x2-x1, y2-y1), interpolation=cv.INTER_LINEAR)
       
        # print(f"size to input: {roi_frame.shape}")
        """roi_frame = cv.resize(roi_frame, (w, h))
        frame[y1:y2, x1:x2] = roi_frame """
        
        scale_x = w/pose.crop_size
        scale_y = h/pose.crop_size

        points = list(filter(lambda x: x is not None, points[4:14:3]))
        points = list(map(lambda x: (x[0] * scale_x + x1, x[1] * scale_y + y1), points))

        x, y, *_ = face_rects[0]

        if wait:
            cv.putText(frame, score, (x - 20, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 6)
            cv.putText(frame, score, (x - 20, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 4)

            if time.time() - game_start_time >= pose.wait_between_rounds:
                wait = 0
                color_to_select, text_color, color_text, time_per_round = init_round(pose, time_per_round * (1 + 1-pose.time_decay), scored)
                game_start_time = time.time()

        # tutorial
        elif pose.tutorial and game:
            prompt = f"Place your hand over the square"
            cv.putText(frame, prompt, (x - 200, y - 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 6)
            cv.putText(frame, prompt, (x - 200, y - 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 4)

            prompt = "of the color the word is"

            cv.putText(frame, prompt, (x - 150, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 6)
            cv.putText(frame, prompt, (x - 150, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 4)
            
            if time.time() - game_start_time >= 5:
                
                wait = 1
                pose.tutorial = 0
                game_start_time = time.time()
        
        # main game
        elif game:
            new_score = 0
            fail = 0
            
            cv.putText(frame, score, (x - 20, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 6)
            cv.putText(frame, score, (x - 20, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 4)

            y1_begin = y1 + (y2 - y1)//4 - pose.box_shape[1] // 2
            y1_end = y1 + (y2 - y1)//4 + pose.box_shape[1] // 2

            y2_begin = y1 + (y2 - y1)//2 - pose.box_shape[1] // 2
            y2_end = y1 + (y2 - y1)//2 + pose.box_shape[1] // 2

            """ postions = [[[x1, y1_begin], [x1 + pose.box_shape[0], y1_end]], 
                [[x2 - pose.box_shape[0], y1_begin], [x2, y1_end]], 
                [[x1, y2_begin], [x1 + pose.box_shape[0], y2_end]], 
                [[x2 - pose.box_shape[0], y2_begin], [x2, y2_end]]]
            
            for color, pos in zip(pose.boxes_colors, postions):    
                cv.rectangle(frame, pos[0], pos[1], color, 6) """

            box_blue = len(list(filter(lambda x: x1 < x[0] < x1 + pose.box_shape[0] and y1_begin < x[1] < y1_end, points)))
            box_green = len(list(filter(lambda x: x2 - pose.box_shape[0] < x[0] < x2 and y1_begin < x[1] < y1_end, points)))
            box_red = len(list(filter(lambda x: x1 < x[0] < x1 + pose.box_shape[0] and y2_begin < x[1] < y2_end, points)))
            box_yellow = len(list(filter(lambda x: x2 - pose.box_shape[0] < x[0] < x2 and y2_begin < x[1] < y2_end, points)))

            if time.time() - game_start_time <= 5:
                y_body_center = (y2 - y1) // 2

                cv.putText(frame, color_text, (x - 20, y_body_center), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 6)
                cv.putText(frame, color_text, (x - 20, y_body_center), cv.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)

            match (color_to_select, box_blue, box_green, box_red, box_yellow):
                case 0, 1, 0, 0, 0:
                    new_score = scored + 1
                case 1, 0, 1, 0, 0:
                    new_score = scored + 1
                case 2, 0, 0, 1, 0:
                    new_score = scored + 1
                case 3, 0, 0, 0, 1:
                    new_score = scored + 1
                    
                case _, 1, 0, 0, 0:
                    fail = 1
                case _, 0, 1, 0, 0:
                    fail = 1
                case _, 0, 0, 1, 0:
                    fail = 1
                case _, 0, 0, 0, 1:
                    fail = 1

                case _:
                    pass

            if new_score:
                scored = new_score
                score = f"Score: {scored}"
                wait = 1
                game_start_time = time.time()

            elif time.time() - game_start_time >= pose.time_per_round or fail or box_blue + box_green + box_red + box_yellow > 1:
                pose.play_prompt = f"Previous score: {scored}"
                restart_game(game, time_per_round, scored, game_start_time, pose)

        else:
            cv.putText(frame, pose.play_prompt, (x - 200, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 6)
            cv.putText(frame, pose.play_prompt, (x - 200, y - 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 4)

            y_begin = y1 + (y2 - y1)//2 - pose.box_shape[1] // 2
            y_end = y1 + (y2 - y1)//2 + pose.box_shape[1] // 2

            x1_center = x1 + pose.box_shape[0] // 2
            x2_center = x2 - pose.box_shape[0] // 2
            
            prompt = "Yes"
            cv.rectangle(frame, (x1, y_begin), (x1 + pose.box_shape[0], y_end), (0, 255, 0, 255), 6)
            cv.putText(frame, prompt, (x1_center - 20, y_begin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 6)
            cv.putText(frame, prompt, (x1_center - 20, y_begin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 4)
            
            prompt = "No"
            cv.rectangle(frame, (x2 - pose.box_shape[0], y_begin), (x2, y_end), (0, 0, 255, 255), 6)
            cv.putText(frame, prompt, (x2_center - 20, y_begin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 255), 6)
            cv.putText(frame, prompt, (x2_center - 20, y_begin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 4)

            box_yes = list(filter(lambda x: x1 < x[0] < x1 + pose.box_shape[0] and y_begin < x[1] < y_end, points))
            box_no = list(filter(lambda x: x2 - pose.box_shape[0] < x[0] < x2 and y_begin < x[1] < y_end, points))

            if len(box_yes) == 1:
                game = 1
                game_start_time = time.time()
            elif len(box_no) == 1 and len(box_yes) == 0:
                cleanup(pose)
                break
        
        """ for t in points:
            cv.circle(frame, (int(t[0]), int(t[1])), 4, (0, 0, 255, 255), -1)
        """
        shifted_frame = linear_transform_img(pose, frame, np.array([dx,dy]), False)

        display_image(shifted_frame, top_left_corner, start_time, game_start_time, time_per_round, pose)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cleanup(pose)
            break
        
if __name__ == "__main__":
    main()
