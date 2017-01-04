#include "clMatrix.h"
#include "clUtils.h"

static __inline void innerCLTransExpandMultiplyMatrices(
		const int leftHeight, const int rightHeight, const int leftWidth,
		netF* leftVals, netF* rightVals, netF* matVals) {

	int totalWidth = leftHeight * rightHeight;
	size_t leftSize = sizeof(float) *  leftWidth * leftHeight;
	size_t rightSize = sizeof(float) * leftWidth * rightHeight;
	size_t matSize = sizeof(float) * totalWidth;

	cl_int err;
	err = clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputA,
		CL_TRUE, 0, leftSize, leftVals, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputB,
		CL_TRUE, 0, rightSize, rightVals, 0, NULL, NULL);

	if (err) {
		printf("Error %d When enqueuing the buffers.\n", err);
	}

	cl_kernel kernel = globalClKernels.transExpandMatrixMult->kernel;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &globalClKernels.inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &globalClKernels.inputB);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &globalClKernels.outputC);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &leftWidth);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &rightHeight);

	if (err) {
		printf("Error %d When setting the arguments.\n", err);
	}

	size_t globalSizes[] = {leftHeight};

	// Execute the kernel over the entire range of the data set
	err = clEnqueueNDRangeKernel(globalClSettings.queue, kernel, 1, NULL,
		globalSizes, NULL, 0, NULL, NULL);
	if (err) {
		printf("Error %d When executing the kernel.\n", err);
	}

	// Wait for the command queue to get serviced before reading back results
	clFinish(globalClSettings.queue);

	clEnqueueReadBuffer(globalClSettings.queue, globalClKernels.outputC, CL_TRUE,
		0, matSize, matVals, 0, NULL, NULL);

	int col;
	for (col = 0; col < totalWidth; col++) {
		matVals[col] /= leftWidth;
	}
}

matrix* gpuTransExpandMultCollapseMatrices(
		matrix* left, matrix *right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if (leftWidth != rightWidth) {
		return NULL;
	}
	int totalWidth = leftHeight * rightHeight;
	matrix* mat = createOrUseSuppliedMatrix(result, 1, totalWidth);
	innerCLTransExpandMultiplyMatrices(leftHeight, rightHeight, leftWidth,
		left->vals, right->vals, mat->vals);

	return mat;
}

static __inline void innerCLTransMultiplyMatrices(const int leftHeight,
		const int leftWidth, const int rightHeight, netF* leftVals,
		netF* rightVals, netF* matVals) {
	size_t totalItems = leftHeight * rightHeight; 
	size_t leftSize = sizeof(float) *  leftWidth * leftHeight;
	size_t rightSize = sizeof(float) * leftWidth * rightHeight;
	size_t matSize = sizeof(float) * totalItems;

	// Write our data set into the input array in device memory
	cl_int err;
	err = clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputA,
		CL_TRUE, 0, leftSize, leftVals, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputB,
		CL_TRUE, 0, rightSize, rightVals, 0, NULL, NULL);
	if (err) {
		printf("Error %d When enqueuing the buffers.\n", err);
	}

	// Set the arguments to our compute kernel
	cl_kernel kernel = globalClKernels.transMatrixMult->kernel;
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &globalClKernels.inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &globalClKernels.inputB);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &globalClKernels.outputC);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &leftHeight);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &leftWidth);
	err |= clSetKernelArg(kernel, 5, sizeof(int), &rightHeight);

	if (err) {
		printf("Error %d When setting the arguments.\n", err);
	}

	size_t globalSizes[] = { leftHeight };


	// Execute the kernel over the entire range of the data set  
	err = clEnqueueNDRangeKernel(globalClSettings.queue, kernel, 1, NULL, globalSizes, NULL, 0, NULL, NULL);
	if (err) {
		printf("Error %d When executing the kernel.\n", err);
	}

	// Wait for the command queue to get serviced before reading back results
	clFinish(globalClSettings.queue);

	// Read the results from the device
	clEnqueueReadBuffer(globalClSettings.queue, globalClKernels.outputC, CL_TRUE,
		0, matSize, matVals, 0, NULL, NULL);
}

matrix* gpuTransMultiplyMatrices(matrix* left, matrix* right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if (leftWidth != rightWidth) {
		return NULL;
	}
	matrix* mat = createOrUseSuppliedMatrix(result, leftHeight, rightHeight);
	innerCLTransMultiplyMatrices(leftHeight, leftWidth, rightHeight, left->vals,
		right->vals, mat->vals);
	return mat;
}