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
            img_out = cv.cvtColor(img_out, cv.COLOR_RGBA2RGB)
            
            plt.imshow(img_out)
            plt.axis('off')
            plt.show()

        sum_square_template = np.array([np.sum(np.square(np.array(x).flatten())) for x in split_img])

        return output, np.mean(sum_square_template[:3])

    except Exception as e:
        print("From subtract_template_avg error:", e)


def correlation_coefficient(input_image, template_minus_avg, pose: PoseEstimator, verbose: bool = False):
    temp_h, temp_w, _ = template_minus_avg.shape

    #input_image = cv.resize(input_image, (640, 480))

    img_h, img_w, _ = input_image.shape

    img_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    # Upload template
    imageIn_template = cl.create_image(
        pose.ctx,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        img_format,
        shape=(temp_w, temp_h),
        hostbuf=template_minus_avg
    )

    # Allocate device-side images
    imageIn_sub_img = cl.Image(pose.ctx, cl.mem_flags.READ_ONLY, img_format, shape=(temp_w, temp_h))
    img_intermediate = cl.Image(pose.ctx, cl.mem_flags.READ_WRITE, img_format, shape=(temp_w, temp_h))

    # Output correlation map
    output_buff = np.zeros(4, dtype=np.ulong)
    buff_out_final = cl.Buffer(pose.ctx, cl.mem_flags.WRITE_ONLY, output_buff.nbytes)
    output_img = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1), dtype=np.ulong)

    kernel1 = pose.kernel_subtract_val_to_img
    kernel2 = pose.kernel_img_mult

    local_work_size = (32, 32)
    global_work_size = ( 
        math.ceil(temp_w / local_work_size[0]) * local_work_size[0],
        math.ceil(temp_h / local_work_size[1]) * local_work_size[1]
    )
    
    for y in range(0, img_h - temp_h + 1, 1):
        for x in range(0, img_w - temp_w + 1, 1):
            sub_img = input_image[y:y+temp_h, x:x+temp_w]           # get sub img
            sub_img_rgba = cv.cvtColor(sub_img, cv.COLOR_BGR2RGBA)  # pass from BGR to RGBA
            sub_img_rgba = sub_img_rgba.astype(np.float32)  # pass from uint8 to float

            # Precompute template average for subtraction
            split_sub_image = cv.split(sub_img_rgba)
            sub_img_pre_calc = np.array([np.sum(np.array(x).flatten())/(temp_w*temp_h) for x in split_sub_image])

            # Update device memory for the current sub-image
            cl.enqueue_copy(pose.commQ, imageIn_sub_img, sub_img_rgba, origin=(0, 0, 0), region=(temp_w, temp_h, 1))

            # Kernel 1: template subtraction
            kernel1.set_arg(0, np.int32(temp_w))
            kernel1.set_arg(1, np.int32(temp_h))
            kernel1.set_arg(2, sub_img_pre_calc)
            kernel1.set_arg(3, imageIn_sub_img)
            kernel1.set_arg(4, img_intermediate)
            cl.enqueue_nd_range_kernel(pose.commQ, kernel1, global_work_size, local_work_size)
            
            # Kernel 2: multiply template_minus_avg with sub_img_rgba_minus_avg
            kernel2.set_arg(0, np.int32(temp_w))
            kernel2.set_arg(1, np.int32(temp_h))
            kernel2.set_arg(2, img_intermediate)
            kernel2.set_arg(3, imageIn_template)
            kernel2.set_arg(4, buff_out_final)
            cl.enqueue_nd_range_kernel(pose.commQ, kernel2, global_work_size, local_work_size)

            # Copy back only the final result
            cl.enqueue_copy(pose.commQ, output_buff, buff_out_final)
            pose.commQ.finish()

            output_img[y, x] = np.mean(output_buff[:3])
    
    if verbose:
        print(output_img)
        """ cv.imshow("frame", output_img)
        cv.waitKey(0) """

    return output_img

def correlation_coefficient_no_attomic(input_image, template_minus_avg, template_sum_mean, pose: PoseEstimator, verbose: bool = False):
    temp_h, temp_w, _ = template_minus_avg.shape

    #input_image = cv.resize(input_image, (640, 480))

    img_h, img_w, _ = input_image.shape

    img_format = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    # Upload template
    imageIn_template = cl.create_image(
        pose.ctx,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        img_format,
        shape=(temp_w, temp_h),
        hostbuf=template_minus_avg
    )

    # Allocate device-side images
    imageIn_sub_img = cl.Image(pose.ctx, cl.mem_flags.READ_ONLY, img_format, shape=(temp_w, temp_h))
    img_intermediate = cl.Image(pose.ctx, cl.mem_flags.READ_WRITE, img_format, shape=(temp_w, temp_h))

    # Output correlation map
    output_sub_img = np.zeros((temp_h ,  temp_w, 4), dtype=np.float32)
    imageOut = cl.create_image(
        pose.ctx,
        cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
        img_format,
        shape=(temp_w, temp_h),
        hostbuf=output_sub_img
    )

    output_img = np.zeros((img_h - temp_h + 1, img_w - temp_w + 1), dtype=np.float16)

    kernel1 = pose.kernel_subtract_val_to_img
    kernel2 = pose.kernel_img_mult_no_attomic

    local_work_size = (32, 32)
    global_work_size = ( 
        math.ceil(temp_w / local_work_size[0]) * local_work_size[0],
        math.ceil(temp_h / local_work_size[1]) * local_work_size[1]
    )
    
    for y in range(0, img_h - temp_h + 1, 1):
        for x in range(0, img_w - temp_w + 1, 1):
            sub_img = input_image[y:y+temp_h, x:x+temp_w]           # get sub img
            sub_img_rgba = cv.cvtColor(sub_img, cv.COLOR_BGR2RGBA)  # pass from BGR to RGBA
            sub_img_rgba = sub_img_rgba.astype(np.float32)  # pass from uint8 to float

            # Precompute template average for subtraction
            split_sub_image = cv.split(sub_img_rgba)
            sub_img_pre_calc = np.array([np.sum(np.array(x).flatten())/(temp_w*temp_h) for x in split_sub_image])
            sum_square_sub = np.array([np.sum(np.square(np.array(x).flatten())) for x in split_sub_image])

            # Update device memory for the current sub-image
            cl.enqueue_copy(pose.commQ, imageIn_sub_img, sub_img_rgba, origin=(0, 0, 0), region=(temp_w, temp_h, 1))

            # Kernel 1: template subtraction
            kernel1.set_arg(0, np.int32(temp_w))
            kernel1.set_arg(1, np.int32(temp_h))
            kernel1.set_arg(2, sub_img_pre_calc)
            kernel1.set_arg(3, imageIn_sub_img)
            kernel1.set_arg(4, img_intermediate)
            cl.enqueue_nd_range_kernel(pose.commQ, kernel1, global_work_size, local_work_size)
            
            # Kernel 2: multiply template_minus_avg with sub_img_rgba_minus_avg
            kernel2.set_arg(0, np.int32(temp_w))
            kernel2.set_arg(1, np.int32(temp_h))
            kernel2.set_arg(2, img_intermediate)
            kernel2.set_arg(3, imageIn_template)
            kernel2.set_arg(4, imageOut)
            cl.enqueue_nd_range_kernel(pose.commQ, kernel2, global_work_size, local_work_size)

            # Copy back only the final result
            cl.enqueue_copy(pose.commQ, output_sub_img, imageOut, origin=(0,0,0), region=(temp_w,temp_h,1))
            pose.commQ.finish()

            split_sub_image = cv.split(output_sub_img)
            flat_minus_avg = np.array([np.sum((np.array(x).flatten())) for x in split_sub_image])

            output_img[y, x] = (np.mean(flat_minus_avg[:3]) / (np.sqrt(template_sum_mean * np.mean(sum_square_sub[:3]))) + 1) * 127.5
    
    if verbose:
        #print(output_img, np.mean(sum_square_sub[:3]))
        output_img = output_img.astype(np.uint8)
        cv.imshow("frame", output_img)
        cv.waitKey(0)

    return output_img