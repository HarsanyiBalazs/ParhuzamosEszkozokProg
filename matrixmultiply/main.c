#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>


const int SAMPLE_SIZE = 49;

int main(void)
{
    int i;
    cl_int err;
    int error_code;
    
    // Get platform
    cl_uint n_platforms;
	cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
		return 0;
	}

    // Get device
	cl_device_id device_id;
	cl_uint n_devices;
	err = clGetDeviceIDs(
		platform_id,
		CL_DEVICE_TYPE_GPU,
		1,
		&device_id,
		&n_devices
	);
	if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
		return 0;
	}

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    const char* kernel_code = load_kernel_source("kernels/sample.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    const char options[] = "-D SET_ME=1234";
    err = clBuildProgram(
        program,
        1,
        &device_id,
        options,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size
        );
        char* build_log = (char*)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size
        );
        // build_log[real_size] = 0;
        printf("Real size : %d\n", real_size);
        printf("Build log : %s\n", build_log);
        free(build_log);
        return 0;
    }
    size_t sizes_param[10];
    size_t real_size;
    err = clGetProgramInfo(
        program,
        CL_PROGRAM_BINARY_SIZES,
        10,
        sizes_param,
        &real_size
    );
    //printf("Real size   : %d\n", real_size);
    //printf("Binary size : %d\n", sizes_param[0]);
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

    // Create the host buffer and initialize it
    float* host_buffer1 = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer1[i] = i;
    }

    float* host_buffer2 = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer2[i] = i+2;
    }

    float* host_buffer3 = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer3[i] = 1;
    }

    // Create the device buffer
    cl_mem device_buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SAMPLE_SIZE * sizeof(float), host_buffer1, &err);
    cl_mem device_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SAMPLE_SIZE * sizeof(float), host_buffer2, &err);
    cl_mem device_buffer3 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SAMPLE_SIZE * sizeof(float), host_buffer3, &err);


    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_buffer2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_buffer3);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&SAMPLE_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    err = clEnqueueWriteBuffer(
        command_queue,
        device_buffer1,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_buffer1,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clEnqueueWriteBuffer1. Error code: %d\n", err);
		return 0;
	}

    err = clEnqueueWriteBuffer(
        command_queue,
        device_buffer2,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_buffer2,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clEnqueueWriteBuffer2. Error code: %d\n", err);
		return 0;
	}

    clEnqueueWriteBuffer(
        command_queue,
        device_buffer3,
        CL_FALSE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_buffer3,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    err = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clEnqueueNDRangeKernel. Error code: %d\n", err);
		return 0;
	}

    clFinish(command_queue);
    // Host buffer <- Device buffer
    err = clEnqueueReadBuffer(
        command_queue,
        device_buffer3,
        CL_TRUE,
        0,
        SAMPLE_SIZE * sizeof(float),
        host_buffer3,
        0,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
		printf("[ERROR] Error calling clEnqueueReadBuffer. Error code: %d\n", err);
		return 0;
	}

    clFinish(command_queue);
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        printf("[%d] = %f, ", i, host_buffer3[i]);
    }

    // Release the resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(host_buffer1);
    free(host_buffer2);
    free(host_buffer3);
}
