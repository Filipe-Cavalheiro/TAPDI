import pyopencl as cl
import numpy as np
import cv2 as cv
import time
import math

def main():
    try:
        plaforms = cl.get_platforms()
        global plaform
        plaform = plaforms[0]
        devices = plaform.get_devices()
        global device
        device = devices[0]
        global ctx
        ctx = cl.Context(devices) # or dev_type=cl.device_type.ALL)
        global commQ
        commQ = cl.CommandQueue(ctx,device)
        file = open("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP6\\prog.c","r")
        global prog
        prog = cl.Program(ctx,file.read())
        prog.build()
    except Exception as e:
        print(e)
        return False

    """ try:
        start_time = time.time()
        # === LOAD IMAGE ===
        img = cv.imread("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP6\\Aula_4_files\\aula4-1.jpg")
        h, w, c = img.shape
        img_rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)

        imageIn = cl.create_image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=(w, h), hostbuf=img_rgba)

        output = np.empty_like(img_rgba)
        out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)

        # === IMAGE PARAMETERS ===
        brightness = 150  # increase brightness
        contrast = 1.2   # increase contrast

        kernel = prog.brightness_and_contrast

        kernel.set_arg(0, np.int32(w))
        kernel.set_arg(1, np.int32(h))
        kernel.set_arg(2, np.int32(0))  # padding
        kernel.set_arg(3, np.int32(brightness))
        kernel.set_arg(4, np.float32(contrast))
        kernel.set_arg(5, imageIn)
        kernel.set_arg(6, out_buf)
        
        width_height_ratio = int(w/h)
        # Launch kernel (1 work-item per pixel)
        local_work_size_var  = [width_height_ratio*32, 32]
        cl.enqueue_nd_range_kernel(commQ, kernel, global_work_size=(math.ceil(w/local_work_size_var[0])*local_work_size_var[0], math.ceil(h/local_work_size_var[1])*local_work_size_var[1]), local_work_size=local_work_size_var)
        cl.enqueue_copy(commQ, output, out_buf)

        # Reshape and show result
        img_out = output.reshape(h, w, 4)
        cv.imshow("Output", img_out)
        print("- gpu: execute --- %s seconds ---" % (time.time() - start_time))
        cv.waitKey(0)
        cv.destroyAllWindows()

    except Exception as e:
        print("Error:", e) """


    try:
        start_time = time.time()
        # === LOAD IMAGE ===
        img = cv.imread("C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP6\\Aula_4_files\\aula4-1.jpg")
        
        img_rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        h, w, c = img_rgba.shape

        fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)

        imageIn = cl.create_image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, fmt, shape=(w, h), hostbuf=img_rgba)

        output = np.empty_like(img_rgba)
        out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)

        kernel = prog.sobel

        kernel.set_arg(0, np.int32(w))
        kernel.set_arg(1, np.int32(h))
        kernel.set_arg(2, imageIn)
        kernel.set_arg(3, out_buf)
        
        local_work_size_var  = [32, 32]
        cl.enqueue_nd_range_kernel(commQ, kernel, global_work_size=(math.ceil(w/local_work_size_var[0])*local_work_size_var[0], math.ceil(h/local_work_size_var[1])*local_work_size_var[1]), local_work_size=local_work_size_var)
        #cl.enqueue_nd_range_kernel(commQ, kernel, global_work_size=[96, 400], local_work_size=[32, 8])
        cl.enqueue_copy(commQ, output, out_buf)
        commQ.finish()

        # Reshape and show result
        img_out = output.reshape(h, w, 4)
        cv.imshow("Output", img_out)
        print("- gpu: execute --- %s seconds ---" % (time.time() - start_time))
        cv.waitKey(0)
        cv.destroyAllWindows()

    except Exception as e:
        print("Error:", e)
    """   
    # === LOAD IMAGE ===
    start_time = time.time()
    img = cv.imread(r"C:\\Users\\caval\\Documents\\Universidade\\9_Semestre\\TAPDI\\TP6\\Aula_4_files\\aula4-1.jpg")

    if img is None:
        raise FileNotFoundError("Image not found. Check the path.")
    
  
    h, w, c = img.shape

    # Convert to BGRA (adds alpha channel)
    #img_rgba = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

    # === APPLY SOBEL ===
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Calculation of Sobelx
    sobelx = cv.Sobel(hsv,cv.CV_64F,1,0,ksize=5)
    
    # Calculation of Sobely
    sobely = cv.Sobel(hsv,cv.CV_64F,0,1,ksize=5)
    
    # Calculation of Laplacian
    laplacian = cv.Laplacian(hsv,cv.CV_64F)

    # === SHOW RESULT ===
    cv.imshow("Adjusted", laplacian)
    print("- cpu: execute --- %s seconds ---" % (time.time() - start_time))
    cv.waitKey(0)
    cv.destroyAllWindows()
    return True """

main()