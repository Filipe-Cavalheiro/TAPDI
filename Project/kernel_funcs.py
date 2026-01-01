import cv2 as cv
import numpy as np
import pyopencl as cl
import math
import matplotlib.pyplot as plt
from PoseEstimator import PoseEstimator

def subtract_template_avg(pose: PoseEstimator, versobe: bool = False):
    face_template = cv.imread(pose.face_template_location)
    face_template_rgba = cv.cvtColor(face_template, cv.COLOR_BGR2RGBA)

    h, w, c = face_template_rgba.shape
    face_template_rgba = face_template_rgba.astype(np.float32)

    # Pre-calculate template averages
    split_img = cv.split(face_template_rgba)
    template_pre_calc = np.array([np.sum(x) / (w * h) for x in split_img])

    try:
        img_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

        # Create OpenCL images
        imageIn = cl.Image(pose.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, img_format, shape=(w, h), hostbuf=face_template_rgba)
        imageOut = cl.Image(pose.ctx, cl.mem_flags.WRITE_ONLY, img_format, shape=(w, h))

        # Set kernel arguments
        kernel = pose.kernel_subtract_val_to_img
        kernel.set_arg(0, np.int32(w))
        kernel.set_arg(1, np.int32(h))
        kernel.set_arg(2, template_pre_calc)
        kernel.set_arg(3, imageIn)
        kernel.set_arg(4, imageOut)

        # Enqueue kernel execution
        cl.enqueue_nd_range_kernel(pose.commQ, kernel, (w, h), None)
        pose.commQ.finish()

        # Allocate host array to copy result
        output = np.zeros((h, w, 4), dtype=np.float32)
        cl.enqueue_copy(pose.commQ, output, imageOut, origin=(0, 0, 0), region=(w, h, 1))
        pose.commQ.finish()

        if versobe:
            img_out = output.astype(np.uint8)
            img_out_rgb = cv.cvtColor(img_out, cv.COLOR_RGBA2BGR)
            
            plt.imshow(img_out_rgb)
            plt.axis('off')
            plt.show()

            img_out_gray = cv.cvtColor(img_out, cv.COLOR_RGBA2GRAY)
            cv.imshow("frame", img_out_gray)
            cv.waitKey(0)

        face_template_gray = cv.cvtColor(face_template, cv.COLOR_RGBA2GRAY)
        sum_square_template = np.sum(np.square(np.array(face_template_gray).flatten()))

        return output, sum_square_template

    except Exception as e:
        print("From subtract_template_avg error:", e)

def normalized_correlation_coefficient(input_image, template_minus_avg, template_sum_mean, pose: PoseEstimator, verbose: bool = False):
    
    template_minus_avg = cv.cvtColor(template_minus_avg, cv.COLOR_RGBA2GRAY)
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    #input_image = cv.resize(input_image, (640, 480))
    temp_h, temp_w = template_minus_avg.shape
    img_h, img_w = input_image.shape

    #images are gray scale
    img_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
    img_format_uint8 = cl.ImageFormat(cl.channel_order.R, cl.channel_type.UNSIGNED_INT8)

    # Upload template
    imageIn_template = cl.create_image(
        pose.ctx,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        img_format,
        shape=(temp_w, temp_h),
        hostbuf=template_minus_avg
    )

    # Allocate device-side images
    imageIn = cl.Image(pose.ctx, cl.mem_flags.READ_ONLY, img_format, shape=(img_w, img_h))
    
    # Output correlation map
    output_buff_shape = [img_h - temp_h + 1 ,  img_w - temp_w + 1]
    output_buff = np.zeros([output_buff_shape[0], output_buff_shape[1]], dtype=np.uint8)
    imageOut = cl.create_image(
        pose.ctx,
        cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
        img_format_uint8,
        shape = (output_buff_shape[1], output_buff_shape[0]),
        hostbuf = output_buff
    )

    kernel = pose.kernel_ncc_tiled

    local_work_size = (8, 8)
    global_work_size = (
        math.ceil((output_buff_shape[1]) / local_work_size[0]) * local_work_size[0],
        math.ceil((output_buff_shape[0]) / local_work_size[1]) * local_work_size[1]
    )


    input_image = input_image.astype(np.float32) 
    # Update device memory for the current sub-image
    cl.enqueue_copy(pose.commQ, imageIn, input_image, origin=(0,0,0), region=(img_w, img_h, 1), is_blocking=True)

    # Kernel: Tiled Normal Correlation Coefficient (ncc_tiled)
    kernel.set_arg(0, np.int32(img_w))
    kernel.set_arg(1, np.int32(img_h))
    kernel.set_arg(2, np.float32(template_sum_mean))
    kernel.set_arg(3, imageIn)
    kernel.set_arg(4, imageIn_template)
    kernel.set_arg(5, imageOut)
    
    cl.enqueue_nd_range_kernel(pose.commQ, kernel, global_work_size, local_work_size)

    cl.enqueue_copy(pose.commQ, output_buff, imageOut, origin=(0,0,0), region=(output_buff_shape[1], output_buff_shape[0], 1))
    pose.commQ.finish()
    
    if verbose:
        print(output_buff)
        cv.imshow("frame", output_buff)
        cv.waitKey(0)

    return output_buff