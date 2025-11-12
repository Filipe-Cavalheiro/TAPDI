
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#JPEG quantification matrix
# receives:
#   - boolean to specify is Y or Cr/Cb component
#   - compression factor (0..100)
def getQuantificationMatrix(LuminanceOrChrominance, compfactor):
    lumQuant = [16,11,10,16,24,40,51,61,
            12,12,14,19,26,58,60,55,
            14,13,16,24,40,57,69,56,
            14,17,22,29,51,87,80,62,
            18,22,37,56,68,109,103,77,
            24,35,55,64,81,104,113,92,
            49,64,78,87,103,121,120,101,
            72,92,95,98,112,100,103,99]

    ChrQuant = [17, 18, 24, 47, 99, 99, 99, 99,
                18, 21, 26, 66, 99, 99, 99, 99,
                24, 26, 56, 99, 99, 99, 99, 99,
                47, 66, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99
                ]

    matrix = np.zeros((8, 8),dtype=float)
    idx = 0
    Quant = lumQuant if LuminanceOrChrominance else ChrQuant
    for y in range(0, 8):
        for x in range(0, 8):
            matrix[y, x] = Quant[idx]*100.0/compfactor
            idx+=1

    return matrix


# Process a 8x8 block image
# receives:
#   - single channel image,
#   - boolean to specify is Y or Cr/Cb component
#   - compression factor (0..100)
def blockProcessing(imgChannelBlock,luminanceOrChrominance, compFactor):
            
    # b)	Convert block to float format and subtract the DC component (128)
    imgChannelBlock = np.float32(imgChannelBlock) - 128

    # c)	Apply the Discrete Cosine Transform (DCT)
    imgChannelBlock = cv.dct(imgChannelBlock)

    # d)	Coefficients Quantization (divide by quantification matrix and round)
    quant_Mat = getQuantificationMatrix(luminanceOrChrominance, compFactor)
    imgChannelBlock = np.divide(imgChannelBlock, quant_Mat)

    # e)	Coefficients rounding (math.round)
    imgChannelBlock = np.round(imgChannelBlock)

    # f) Coefficients recovering
    imgChannelBlock = np.multiply(imgChannelBlock, quant_Mat)

    # g)	Apply the Discrete Cosine Inverse Transform
    imgChannelBlock = cv.idct(imgChannelBlock)

    # h) Add DC component, clip to 0..255 and convert to byte
    imgChannelBlock += 128
    imgChannelBlock = np.clip(imgChannelBlock, 0, 255)
    result = np.uint8(imgChannelBlock)

    return result

def main():
    COMPRESSION_PER =  30.0
    img_BGR = cv.imread("./usb_32x32.png")
    img_BGR = cv.cvtColor(img_BGR, cv.COLOR_BGR2RGB)

    img_YCrCb = cv.cvtColor(img_BGR, cv.COLOR_BGR2YCrCb)
    imgChannelY, imgChannelCr, imgChannelCb  = cv.split(img_YCrCb)

    imgChannelCb = cv.resize(imgChannelCb, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)
    imgChannelCr = cv.resize(imgChannelCr, None, fx=0.5, fy=1, interpolation=cv.INTER_AREA)

    BLOCK_SIZE = 8

    height, width = imgChannelY.shape
    N_width_blocks = int(width/ BLOCK_SIZE)
    N_height_blocks = int(height/ BLOCK_SIZE)
    reconstructedY = np.zeros_like(imgChannelY)
    for i in range(N_height_blocks):
        for j in range(N_width_blocks):
            imgChannelBlock = imgChannelY[i*BLOCK_SIZE : i*BLOCK_SIZE+BLOCK_SIZE, j*BLOCK_SIZE : j*BLOCK_SIZE+BLOCK_SIZE]
            resultBlock = blockProcessing(imgChannelBlock,1, COMPRESSION_PER)
            reconstructedY[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE, j*BLOCK_SIZE : (j+1)*BLOCK_SIZE] = resultBlock

    height, width = imgChannelCb.shape
    N_width_blocks = int(width/ BLOCK_SIZE)
    N_height_blocks = int(height/ BLOCK_SIZE)
    reconstructedCb = np.zeros_like(imgChannelCb)
    for i in range(N_height_blocks):
        for j in range(N_width_blocks):
            imgChannelBlock = imgChannelCb[i*BLOCK_SIZE : i*BLOCK_SIZE+BLOCK_SIZE, j*BLOCK_SIZE : j*BLOCK_SIZE+BLOCK_SIZE]
            resultBlock = blockProcessing(imgChannelBlock,0, COMPRESSION_PER)
            reconstructedCb[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE, j*BLOCK_SIZE : (j+1)*BLOCK_SIZE] = resultBlock

    height, width = imgChannelCr.shape
    N_width_blocks = int(width/ BLOCK_SIZE)
    N_height_blocks = int(height/ BLOCK_SIZE)
    reconstructedCr = np.zeros_like(imgChannelCr)
    for i in range(N_height_blocks):
        for j in range(N_width_blocks):
            imgChannelBlock = imgChannelCr[i*BLOCK_SIZE : i*BLOCK_SIZE+BLOCK_SIZE, j*BLOCK_SIZE : j*BLOCK_SIZE+BLOCK_SIZE]
            resultBlock = blockProcessing(imgChannelBlock,0, COMPRESSION_PER)
            reconstructedCr[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE, j*BLOCK_SIZE : (j+1)*BLOCK_SIZE] = resultBlock

    reconstructedCb = cv.resize(reconstructedCb ,None,fx=2, fy=1, interpolation=cv.INTER_AREA )
    reconstructedCr = cv.resize(reconstructedCr ,None,fx=2, fy=1, interpolation=cv.INTER_AREA )
    img_JPEG = cv.merge((reconstructedY, reconstructedCr, reconstructedCb))
    img_JPEG = cv.cvtColor(img_JPEG, cv.COLOR_YCrCb2BGR)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_BGR)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(img_JPEG)
    plt.axis('off')
    plt.show()

    

if __name__ == "__main__":
    main()
