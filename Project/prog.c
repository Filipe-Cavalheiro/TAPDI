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
    __global atomic_ulong  *imageOut
)
{
    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= w || y >= h) return;

    float4 p_1 = read_imagef(imageIn_1, sampler, (int2)(x, y));
    float4 p_2 = read_imagef(imageIn_2, sampler, (int2)(x, y));
    ulong4 p_1_long = convert_ulong4(p_1);
    ulong4 p_2_long = convert_ulong4(p_2);
    atomic_add(&imageOut[0], p_1_long.x * p_2_long.x);
    atomic_add(&imageOut[1], p_1_long.y * p_2_long.y);
    atomic_add(&imageOut[2], p_1_long.z * p_2_long.z);
    atomic_add(&imageOut[3], p_1_long.w * p_2_long.w);
}

__kernel void img_mult_pix2pix_no_attomic(
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