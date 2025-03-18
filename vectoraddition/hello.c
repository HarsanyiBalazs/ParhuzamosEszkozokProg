#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

const char* kernel_code =
    "__kernel void hello_kernel(__global float* buffer1,__global float* buffer2,__global float* buffer3, int n) {\n"
    "   buffer3[get_global_id(0)] = buffer1[get_global_id(0)]+buffer2[get_global_id(0)];\n"
    "}\n"
;

/*
    "   if (get_global_id(0) < n) {\n"
    "       buffer[get_global_id(0)] = 11;\n"
    "   }\n"
*/

/*
const char* kernel_code =
    "__kernel void hello_kernel(__global int* buffer, int n) {\n"
    "   if (get_global_id(0) < n) {\n"
    "       buffer[get_global_id(0)] = get_global_id(0) * 10;\n"
    "   }\n"
    "}\n"
;
*/

/*
const char* kernel_code =
    "__kernel void hello_kernel(__global int* buffer, int n) {\n"
    "   if (get_global_id(0) % 2 == 0) {\n"
    "       buffer[get_global_id(0)] = 11;\n"
    "   } else {\n"
    "       buffer[get_global_id(0)] = 22;\n"
    "   }\n"
    "}\n"
;*/


const int SAMPLE_SIZE = 1000;

int main(void)
{
    int i;
    cl_int err;

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
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        return 0;
    }
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

    // Create the host buffer and initialize it
    float* host_buffer1 = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer1[i] = i;
    }

    float* host_buffer2 = (float*)malloc(SAMPLE_SIZE * sizeof(float));
    for (i = 0; i < SAMPLE_SIZE; ++i) {
        host_buffer2[i] = i*2;
    }

    float* host_buffer3 = (float*)malloc(SAMPLE_SIZE * sizeof(float));

    // Create the device buffer
    cl_mem device_buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(float), NULL, NULL);
    cl_mem device_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(float), NULL, NULL);
    cl_mem device_buffer3 = clCreateBuffer(context, CL_MEM_READ_WRITE, SAMPLE_SIZE * sizeof(float), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_buffer2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_buffer3);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&SAMPLE_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
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

    clEnqueueWriteBuffer(
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

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (SAMPLE_SIZE + local_work_size + 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    clEnqueueNDRangeKernel(
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

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
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

    for (i = 0; i < SAMPLE_SIZE; ++i) {
        printf("[%d] = %d, ", i, host_buffer3[i]);
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
