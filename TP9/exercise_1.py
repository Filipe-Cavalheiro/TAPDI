import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def main():
    # ex1
    # All the 6 methods for comparison in a list
    methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
                'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
 

    template = cv.imread('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\car template (1).bmp')
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    
    # Assign directory
    directory = r'C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\road'

    # Iterate over files in directory
    for name in os.listdir(directory):
        print('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\road\\' + str(name))

        original_img = cv.imread('C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP9\\aula9\\road\\' + str(name))
        assert original_img is not None, "file could not be read, check with os.path.exists()"
    
        for meth in methods:
            img = original_img.copy()
            method = getattr(cv, meth)
        
            # Apply template Matching
            res = cv.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
        
            cv.rectangle(img,top_left, bottom_right, 255, 2)
        
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
        
            plt.show()


if __name__ == "__main__":
    main()