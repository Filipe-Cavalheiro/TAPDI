__kernel void power2(__global int* arr)
{
    int i = get_global_id(0);
    if (i < 25){
        arr[i] = arr[i] * arr[i];
    }
}

__kernel void power_const(__global int* arr, int K, __global int* result)
{
    int i = get_global_id(0);
    if (i > 25)
        return;
    arr[i] = arr[i] * K;
    result[0] += arr[i];
}

__kernel void negative(__global uchar* image, int w, int h, int padding, __global uchar* imageOut)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (w*3 + padding) + x*3 ;
    if ((x < w) && (y < h)) { // check if x and y are valid image coordinates
        imageOut[idx] = 255 - image[idx];
        imageOut[idx+1] = 255 - image[idx+1];
        imageOut[idx+2] = 255 - image[idx+2];
    }
}


__kernel void brightness_and_contrast(int w, int h, int padding, int brightness, float contrast, __read_only image2d_t imageIn, __global uchar4* imageOut)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP |
                              CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 pixel = read_imagef(imageIn, sampler, (int2)(x, y)) * 255.0f;

    pixel.x = clamp(mad(contrast, pixel.x, brightness), 0.0f, 255.0f);
    pixel.y = clamp(mad(contrast, pixel.y, brightness), 0.0f, 255.0f);
    pixel.z = clamp(mad(contrast, pixel.z, brightness), 0.0f, 255.0f);
    pixel.w = 255.0f;

    int idx = y * w + x;
    imageOut[idx] = (uchar4)(pixel.x, pixel.y, pixel.z, pixel.w);
}

__kernel void sobel(int w, int h,
                    __read_only image2d_t imageIn,
                    __global uchar4* imageOut)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    
    // Sobel kernels
    int Gx[3][3] = {{1, 0, -1},
                    {2, 0, -2},
                    {1, 0, -1}};
    int Gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}};

    uint4 gx = 0.0f, gy = 0.0f;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            uint4 p = read_imageui(imageIn, sampler, (int2)(x + j, y + i));
            gx += p * Gx[i+1][j+1];
            gy += p * Gy[i+1][j+1];
        }
    }

    float4 g = sqrt(convert_float4(gx*gx + gy*gy));
    g = clamp(g, 0.0f, 255.0f);
    uint4 g_uint = convert_uint4(g); 
    
    int idx = y * w + x;
    imageOut[idx] = (uchar4)(g_uint.x, g_uint.y, g_uint.z, 255); 
}
   

__kernel void subtract_val_to_img(
    int w,
    int h,
    float4 val_to_sub,
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    float4 p = read_imagef(imageIn, sampler, (int2)(x, y));
    write_imagef(imageOut, (int2)(x, y), p - val_to_sub);
}

__kernel void img_mult_pix2pix(
    int w, int h,
    __read_only image2d_t imageIn_1,
    __read_only image2d_t imageIn_2,
    __write_only image2d_t out_put_mult
)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= w || y >= h)
        return;

    float4 p_1 = read_imagef(imageIn_1, sampler, (int2)(x, y));
    float4 p_2 = read_imagef(imageIn_2, sampler, (int2)(x, y));

    float4 result = p_1 * p_2;

    write_imagef(out_put_mult, (int2)(x, y), result);
}

#define LOCAL_H 8
#define LOCAL_W 8
#define TEMP_H 105
#define TEMP_W 90

__kernel void ncc_tiled(
    int img_w,
    int img_h,
    float template_sum_mean,
    __read_only image2d_t inputImage,
    __read_only image2d_t templateImage_min_avg,
    __write_only image2d_t outputImage) 
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    const int group_x = get_group_id(0) * LOCAL_W;
    const int group_y = get_group_id(1) * LOCAL_H;

    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    // Shared tile in local memory
    __local float tile[LOCAL_H + TEMP_H - 1][LOCAL_W + TEMP_W - 1];

    /* Load tile */
    for (int y = local_y; y < LOCAL_H + TEMP_H - 1; y += LOCAL_H) {
        for (int x = local_x; x < LOCAL_W + TEMP_W - 1; x += LOCAL_W) {
            int2 coord = (int2)(group_x + x, group_y + y);
            float4 pix = read_imagef(inputImage, sampler, coord);
            tile[y][x] = pix.x; // take only the first channel (grayscale)
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Skip out-of-bounds work-items */
    if (global_x >= img_w - TEMP_W + 1 || global_y >= img_h - TEMP_H + 1) return;

    // Compute sub-image average
    float sub_img_avg = 0.0f;
    float sub_img_sum_square = 0.0f;
    for (int template_y = 0; template_y < TEMP_H; ++template_y) {
        for (int template_x = 0; template_x < TEMP_W; ++template_x) {
            sub_img_avg += tile[local_y + template_y][local_x + template_x];
            sub_img_sum_square += tile[local_y + template_y][local_x + template_x]*tile[local_y + template_y][local_x + template_x];
        }
    }
    sub_img_avg /= (TEMP_H * TEMP_W);

    // Compute correlation
    float correlation = 0.0f;
    for (int template_y = 0; template_y < TEMP_H; ++template_y) {
        for (int template_x = 0; template_x < TEMP_W; ++template_x) {
            float sub_pix = tile[local_y + template_y][local_x + template_x];
            float tmp_min_avg = read_imagef(templateImage_min_avg, sampler, (int2)(template_x, template_y)).x;
            correlation += (sub_pix - sub_img_avg) * tmp_min_avg;
        }
    }

    correlation = correlation/sqrt(template_sum_mean * sub_img_sum_square);
    correlation = (correlation + 1) * 127.5f;

    write_imageui(outputImage, (int2)(global_x, global_y), (uint4)((uint)correlation, 0, 0, 255));
}
