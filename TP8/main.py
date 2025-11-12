import ImageFFT as ifft
import python_files.imageForms as IF
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import ImageDeconvolution as iD

def main():
    # ex1
    doggo = cv.imread("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP8\\aula8_1.bmp")
    doggo_gray = cv.cvtColor(doggo, cv.COLOR_BGR2GRAY)
    mag_doggo_gray, phase_doggo_gray = ifft.GetFFT_Mag_Phase(doggo_gray)
    # IF.showSideBySideImages(np.fft.fftshift(cv.log(mag_doggo_gray)), np.fft.fftshift(phase_doggo_gray), 'FFT')

    # ex2
    birdy = cv.imread("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP8\\aula8_2.bmp")
    birdy_gray2 = cv.cvtColor(birdy, cv.COLOR_BGR2GRAY)
    mag_birdy_gray2, phase_birdy_gray = ifft.GetFFT_Mag_Phase(birdy_gray2)
    new_img = ifft.GetFFT_Inverse_Mag_Phase(mag_doggo_gray, phase_birdy_gray)
    """     plt.imshow(new_img, 'gray')
    plt.show()
    new_img2 = ifft.GetFFT_Inverse_Mag_Phase(mag_birdy_gray2, phase_doggo_gray)
    plt.imshow(new_img2, 'gray')
    plt.show() """

    # ex3
    mask_img = ifft.CreateFilterMask_Ideal(mag_doggo_gray.shape, 30, True)
    output = mag_doggo_gray.copy()
    cv.multiply(mag_doggo_gray, np.fft.fftshift(mask_img), output)
    new_img = ifft.GetFFT_Inverse_Mag_Phase(output, phase_doggo_gray)
    """plt.imshow(new_img, 'gray')
    plt.show()"""

    #ex4
    mask_img = ifft.CreateFilterMask_Gaussian(mag_doggo_gray.shape, 30, True)
    output = mag_doggo_gray.copy()
    cv.multiply(mag_doggo_gray, np.fft.fftshift(mask_img), output)
    new_img = ifft.GetFFT_Inverse_Mag_Phase(output, phase_doggo_gray)
    """plt.imshow(new_img, 'gray')
    plt.show()"""

    #ex5
    square = cv.imread("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP8\\aula8_3.bmp")
    filter = iD.GetFilterConv(True, 10)
    bad_vision_square = cv.filter2D(square, cv.CV_8U, filter, anchor=(np.int32(filter.shape[0]/2), np.int32(filter.shape[1]/2)))
    """ plt.imshow(bad_vision_square, 'gray')
    plt.show()  """

    #ex6
    noiseAmp = 2
    attomic_bomb_filter = np.random.rand(square.shape[0], square.shape[1]) * noiseAmp
    """ plt.imshow(doggo_gray+attomic_bomb_filter, 'gray')
    plt.show() """

    #ex7
    bad_vision_square_gray = cv.cvtColor(bad_vision_square, cv.COLOR_BGR2GRAY)
    square_restored = iD.InverseDeconvolutionWiener(bad_vision_square_gray+attomic_bomb_filter, 0.00001 , True, 20)
    plt.imshow(square_restored, 'gray')
    plt.show() 



if __name__ == "__main__":
    main()