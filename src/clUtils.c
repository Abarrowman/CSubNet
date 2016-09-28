#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <math.h>
#include <stdlib.h>
#include "utils.h"
#include "clUtils.h"

#define CL_UTILS_INCLUDE_DEBUGS 1


clSettings globalClSettings = {.initialized  = 0};
clKernels globalClKernels;



static const char *addKernelSrc =
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
"\n";


static const char* transMatrixMultSrc =
"__kernel void transMatrixMult(__global const float *left, __global const float *right, __global float *target, const int leftHeight, const int leftWidth, const int rightHeight) { \n"
"  int row = get_global_id(0);\n"
"  int col = get_global_id(1);\n"
"\n"
"  __global float* a = left + row * leftWidth;\n"
"  __global float* b = right + leftWidth * col;\n"
"  float sum = 0;\n"
"  int n;\n"
"  for(n = 0; n < leftWidth; n++) {\n"
"    sum += left[row * leftWidth + n] * right[leftWidth * col + n];\n"
"  }\n"
"  target[row * rightHeight + col] = sum;\n"
"}";

static const char* transExpandMatrixMultSrc = "__kernel void transExpandMatrixMult(__global const float *left, __global const float *right, __global float *target,\n"
"  const int leftWidth, const int rightHeight) { \n"
"  int leftRow = get_global_id(0);\n"
"  int rightRow = get_global_id(1);\n"
"\n"
"  float sum = 0;\n"
"  int n;\n"
"  for(n = 0; n < leftWidth; n++) {\n"
"    sum += left[leftRow * leftWidth + n] * right[rightRow * leftWidth + n];\n"
"  }\n"
"  target[leftRow * rightHeight + rightRow] = sum;\n"
"}";

/*const char* transMatrixMultSrc =
"__kernel void transMatrixMult(__global const float *left, __global const float *right, __global float *target, const int leftHeight, const int leftWidth, const int rightHeight) { \n"
"  int id = get_global_id(0);\n"
"  if (id < (leftHeight * rightHeight)) {\n"
"    int row = id / rightHeight;\n"
"    int col = id % rightHeight;\n"
"    __global float* a = left + row * leftWidth;\n"
"    __global float* b = right + leftWidth * col;\n"
"    float sum = 0;\n"
"    int n;\n"
"    for(n = 0; n < leftWidth; n++) {\n"
"      sum += a[n] * b[n];\n"
"    }\n"
"    target[id] = sum;\n"
"  }\n"
"}\n"
"\n";*/

/*const char* transMatrixMultSrc =
"__kernel void transMatrixMult(__global const float *left, __global const float *right, __global float *target, const int leftHeight, const int leftWidth, const int rightHeight) { \n"
"  int id = get_global_id(0);\n"
"  if (id < (leftHeight * rightHeight)) {\n"
"    target[id] = 5.0;\n"
"  }"
"}\n"
"\n";*/

int clCoreInit() {
	if (globalClSettings.initialized) {
		return 0;
	}
	cl_uint count;
	cl_int err;
	err = clGetPlatformIDs(1, NULL, &count);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while determining how many OpenCL platforms there are.\n", err);
		return 1;
	}
	if (count < 1) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "There are no OpenCL platforms.\n");
		return 1;
	}

	//for now just use the first platform
	cl_platform_id clPlatform;
	err = clGetPlatformIDs(1, &clPlatform, NULL);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while getting the first platform id.\n", err);
		return 1;
	}


	cl_device_id deviceIds[8];
	err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 8, deviceIds, &count);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while getting the device ids.\n", err);
		return 1;
	}
	if (count < 1) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "There are no OpenCL devices.\n");
		return 1;
	} else if (count > 8) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "There are more than 8 OpenCL devices, only the first 8 will be considered.\n");
	}

	char paramBuffer[1024];
	size_t longest = -1;
	int bestIdx = 0;
	int n;
	for (n = 0; n < count; n++) {
		size_t paramSize;
		err = clGetDeviceInfo(deviceIds[n], CL_DEVICE_EXTENSIONS, 1024, paramBuffer, &paramSize);
		//PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Device %d Extensions: %.*s\n", n, (int)paramSize, paramBuffer);
		//the device with the most extensions is likely to be the best
		if (paramSize > longest) {
			longest = paramSize;
			bestIdx = n;
		}
		if (err) {
			PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while getting device info.\n", err);
			return 1;
		}
	}
	//for now use only the best device
	cl_device_id chosenDevice = deviceIds[bestIdx];
	cl_context context = clCreateContext(NULL, 1, &chosenDevice, NULL, NULL, &err);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while creating an OpenCL context.\n", err);
		return 1;
	}

	#ifdef __APPLE__
	cl_command_queue queue = clCreateCommandQueue(context, chosenDevice, 0, &err);
	#else
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, chosenDevice, 0, &err);
	#endif
	if (err) {
		printf("Error %d occured when creating a command queue.\n", err);
		clReleaseContext(context);
		return 1;
	}

	globalClSettings.platform = clPlatform;
	globalClSettings.device = chosenDevice;
	globalClSettings.context = context;
	globalClSettings.queue = queue;
	globalClSettings.initialized = 1;
	return 0;
}

void clCoreEnd() {
	if (globalClSettings.initialized) {
		globalClSettings.initialized = 0;
		cl_int err;
		err = clReleaseCommandQueue(globalClSettings.queue);
		if (err) {
			PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while releasing OpenCL command queue.\n", err);
		}
		err = clReleaseContext(globalClSettings.context);
		if (err) {
			PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while releasing OpenCL context.\n", err);
		}
	}
}

int clInit() {
	int result = clCoreInit();
	if (result) {
		return result;
	}
	clStandAloneKernel* transMatrixMult = createStandAloneKernel(transMatrixMultSrc, "transMatrixMult");
	clStandAloneKernel* transExpandMatrixMult = createStandAloneKernel(transExpandMatrixMultSrc, "transExpandMatrixMult");
	if (transMatrixMult == NULL) {
		clCoreEnd();
		return 1;
	}
	globalClKernels.transMatrixMult = transMatrixMult;
	globalClKernels.transExpandMatrixMult = transExpandMatrixMult;

	size_t maxSize = sizeof(float) * 4000 * 4000;
	globalClKernels.inputA = clCreateBuffer(globalClSettings.context, CL_MEM_READ_ONLY, maxSize, NULL, NULL);
	globalClKernels.inputB = clCreateBuffer(globalClSettings.context, CL_MEM_READ_ONLY, maxSize, NULL, NULL);
	globalClKernels.outputC = clCreateBuffer(globalClSettings.context, CL_MEM_WRITE_ONLY, maxSize, NULL, NULL);
	return 0;
}

void clEnd() {
	clReleaseMemObject(globalClKernels.inputA);
	clReleaseMemObject(globalClKernels.inputB);
	clReleaseMemObject(globalClKernels.outputC);
	deleteStandAloneKernel(globalClKernels.transMatrixMult);
	deleteStandAloneKernel(globalClKernels.transExpandMatrixMult);
	clCoreEnd();
}

clStandAloneKernel* createStandAloneKernel(const char* src, const char* name) {
	cl_int err;
	cl_program program = clCreateProgramWithSource(globalClSettings.context, 1, &src, NULL, &err);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured when creating OpenCL program.\n", err);
		return NULL;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured when building the program.\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			size_t log_size;
			clGetProgramBuildInfo(program, globalClSettings.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			char *log = (char *)malloc(log_size);
			clGetProgramBuildInfo(program, globalClSettings.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

					PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS,  "%.*s\n", 	(int)log_size, log);


			free(log);
			
		}
		err = clReleaseProgram(program);
		return NULL;
	}

	// Create the compute kernel in the program we wish to run
	cl_kernel kernel = clCreateKernel(program, name, &err);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured when creating the kernel.\n", err);
		err = clReleaseProgram(program);
		return NULL;
	}

	clStandAloneKernel* stand = malloc(sizeof(clStandAloneKernel));
	stand->kernel = kernel;
	stand->program = program;
	return stand;
}

void deleteStandAloneKernel(clStandAloneKernel* kernel) {
	cl_int err;
	err = clReleaseKernel(kernel->kernel);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while releasing OpenCL kernel.\n", err);
	}
	err = clReleaseProgram(kernel->program);
	if (err) {
		PRINT_FLUSH(CL_UTILS_INCLUDE_DEBUGS, "Error %d occured while releasing OpenCL program.\n", err);
	}
}

void clTest() {
	int n = 100000;

	float *h_a;
	float *h_b;
	float *h_c;

	// Device input buffers
	cl_mem d_a;
	cl_mem d_b;
	cl_mem d_c;

	// Size, in bytes, of each vector
	size_t bytes = n*sizeof(float);

	// Allocate memory for each vector on host
	h_a = (float*)malloc(bytes);
	h_b = (float*)malloc(bytes);
	h_c = (float*)malloc(bytes);

	// Initialize vectors on host
	int i;
	for (i = 0; i < n; i++)
	{
		float iAsF = (float)i;
		h_a[i] = sinf(iAsF)*sinf(iAsF);
		h_b[i] = cosf(iAsF)*cosf(iAsF);
	}

	size_t globalSize, localSize;
	cl_int err;

	// Number of work items in each local work group
	localSize = 64;

	// Number of total work items - localSize must be devisor
	globalSize = (size_t)(ceilf(n / (float)localSize) * localSize);

	clStandAloneKernel* stand = createStandAloneKernel(addKernelSrc, "vecAdd");

	// Create the input and output arrays in device memory for our calculation
	d_a = clCreateBuffer(globalClSettings.context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_b = clCreateBuffer(globalClSettings.context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_c = clCreateBuffer(globalClSettings.context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(globalClSettings.queue, d_a, CL_TRUE, 0,
		bytes, h_a, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(globalClSettings.queue, d_b, CL_TRUE, 0,
		bytes, h_b, 0, NULL, NULL);
	if (err) {
		printf("Error %d When enqueuing the buffers.\n", err);
	}

	// Set the arguments to our compute kernel
	err = clSetKernelArg(stand->kernel, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(stand->kernel, 1, sizeof(cl_mem), &d_b);
	err |= clSetKernelArg(stand->kernel, 2, sizeof(cl_mem), &d_c);
	err |= clSetKernelArg(stand->kernel, 3, sizeof(int), &n);
	if (err) {
		printf("Error %d When setting the arguments.\n", err);
	}

	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(globalClSettings.queue, stand->kernel, 1, NULL, &globalSize, &localSize,
		0, NULL, NULL);
	if (err) {
		printf("Error %d When executing the kernel.\n", err);
	}

	// Wait for the command queue to get serviced before reading back results
	clFinish(globalClSettings.queue);

	// Read the results from the device
	clEnqueueReadBuffer(globalClSettings.queue, d_c, CL_TRUE, 0,
		bytes, h_c, 0, NULL, NULL);

	//Sum up vector c and print result divided by n, this should equal 1 within error
	float sum = 0;
	for (i = 0; i<n; i++)
		sum += h_c[i];

	// release OpenCL resources
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);

	deleteStandAloneKernel(stand);

	//release host memory
	free(h_a);
	free(h_b);
	free(h_c);

	printf("final result: %f count: %d avg: %f\n", sum, n, sum / n);
}