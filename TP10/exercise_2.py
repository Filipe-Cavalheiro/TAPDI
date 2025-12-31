import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from imageOpticalFlow import LucasKanade_OF, Farneback_OF
import os

def main():
    # Assign directory
    directory = r'Aula_10_files/Aula_10_video/'


    # Iterate over files in directory
    for name in os.listdir(directory):
        LucasKanade_OF(directory + name, 3)
        Farneback_OF(directory + name)

if __name__ == "__main__":
    main()