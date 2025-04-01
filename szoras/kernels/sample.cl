__kernel void hello_kernel(__global float* buffer1, float atlag, __global float* buffer2, int n) {
    buffer2[get_global_id(0)] = (buffer1[get_global_id(0)] - atlag)*(buffer1[get_global_id(0)] - atlag);
}
