__kernel void hello_kernel(__global float* buffer1,__global float* buffer2,__global float* buffer3, int n) {
    int dimension=(int)sqrt( (float)n );
    int i = get_global_id(0);
    for(int j=0;j<dimension;j++) {
        for (int k=0;k<dimension;k++) {
            buffer3[i * dimension + j] += buffer1[i * dimension + k] * buffer2[k * dimension + j];
        }
    }
}

//buffer3[i * dimension + j] += buffer1[i * dimension + k] * buffer2[k * dimension + j];