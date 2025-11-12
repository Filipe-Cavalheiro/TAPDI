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