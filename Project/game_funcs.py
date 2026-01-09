import time
import cv2 as cv
import random
from gameManager import GameManager
import numpy as np

def restart_game(game, time_per_round, scored, game_start_time, gm: GameManager):
    game = 0
    scored = 0
    time_per_round = gm.time_per_round
    game_start_time = 0
    return game, scored, time_per_round, game_start_time

def display_image(img, text_pos, start_time, game_start_time, time_per_round, gm: GameManager):
    fps = 1/(time.time() - start_time)

    prompt = f"fps: {fps:.2f}"
    cv.putText(img, prompt, text_pos, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 3)
    cv.putText(img, prompt, text_pos, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 2)
    
    if (game_start_time != 0):
        prompt = f"time_left: {(time_per_round + game_start_time - time.time()):.2f}"
        cv.putText(img, prompt, (gm.img_width - 250, text_pos[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 3)
        cv.putText(img, prompt, (gm.img_width - 250, text_pos[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 2)
    
    cv.imshow("frame", img)

def init_round(gm: GameManager, time_per_round, scored) -> list[int]:
    color_to_select = random.randint(0, len(gm.game_colors) - 1)
    text_color = (255 * int(color_to_select == 0), 255 * int(color_to_select == 1 or color_to_select == 3), 255 * int(color_to_select == 2 or color_to_select == 3), 255)

    prob = random.randint(0, 100)
    color_text = None

    if prob <= 10/np.emath.logn(100, scored + 1):
        color_text = gm.game_colors[color_to_select]
    else:
        temp_colors = [val for i, val in enumerate(gm.game_colors) if i != color_to_select]
        color_text = random.choice(temp_colors)
        
    #np.random.shuffle(gm.boxes_colors)

    return color_to_select, text_color, color_text, time_per_round * gm.time_decay if time_per_round > 2 else time_per_round
