#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <math.h>
#include "matrix.h"
#include "utils.h"
#include "clUtils.h"

//#include <intrin.h>
//#include <immintrin.h>


matrix* initMatrix(matrix* mat, int height, int width) {
	mat->vals = (netF*)malloc(sizeof(netF) * width * height);
	//mat->vals = (netF*)_aligned_malloc(sizeof(netF) * width * height, 16);
	mat->width = width;
	mat->height = height;
	return mat;
}

matrix* createMatrix(int height, int width) {
	return initMatrix((matrix*)malloc(sizeof(matrix)), height, width);
}

matrix* createIdentityMatrix(int height, int width) {
	return fillMatrixIdentity(createMatrix(height, width));
}

matrix* createMatrixWithValues(int height, int width, netF* vals) {
	return setMatrixValues(createMatrix(height, width), vals);
}

matrix* fillMatrixZero(matrix* m) {
	memset(m->vals, 0, sizeof(netF) * m->width * m->height);
	return m;
}

matrix* fillMatrixRandom(matrix* m) {
	int i;
	for (i = 0; i < (m->width * m->height); i++) {
		m->vals[i] = randomNetF();
	}
	return m;
}

matrix* fillMatrixIdentity(matrix* m) {
	int minDim = (m->width < m->height) ? m->width : m->height;
	int idx;
	fillMatrixZero(m);
	for (idx = 0; idx < minDim; idx++) {
		m->vals[idx * m->width + idx] = 1;
	}
	return m;
}

void clearMatrix(matrix* mat) {
	free(mat->vals);
	mat->vals = NULL;
}

void deleteMatrix(matrix* mat) {
	clearMatrix(mat);
	free(mat);
}


void writeMatrixRow(FILE *stream, matrix* mat, char colSep, char rowSep, int row) {
	int lastCol = mat->width - 1;
	int col;
	for (col = 0; col <lastCol;col++) {
		fprintf(stream, "%f%c", mat->vals[row * mat->width + col], colSep);
	}
	fprintf(stream, "%f%c", mat->vals[row * mat->width + col], rowSep);
}

matrix* writeMatrix(FILE *stream, matrix* mat, char colSep, char rowSep) {
	int row;
	for (row = 0; row < mat->height; row++) {
		writeMatrixRow(stream, mat, colSep, rowSep, row);
	}
	fflush(stream);
	return mat;
}

matrix* printMatrix(matrix* mat) {
	return writeMatrix(stdout, mat, ' ', '\n');
}

matrix* setMatrixVal(matrix* mat, int row, int col, netF val) {
	if ((col < mat->width) && (row < mat->height)) {
		mat->vals[row * mat->width + col] = val;
	}
	return mat;
}

matrix* setMatrixValues(matrix *mat, netF *dubs) {
	memcpy(mat->vals, dubs, sizeof(netF) * mat->width * mat->height);
	return mat;
}

netF* getMatrixVal(matrix* mat, int row, int col) {
	return &(mat->vals[mat->width * row + col]);
}

static matrix* createOrUseSuppliedMatrix(matrix* supplied, int height, int width) {
	if (supplied == NULL) {
		return createMatrix(height, width);
	} else if ((supplied->width != width) || (supplied->height != height)) {
		PRINT_FLUSH(1, "Supplied matrix is of the wrong size %dx%d instead of %dx%d",
			supplied->height, supplied->width, height, width);
		exit(1);
		return NULL;
	} else {
		return supplied;
	}
}

static __inline void innerMultiplyMatrices(const int leftHeight, const int leftWidth, const int rightWidth, netF* leftVals, netF* rightVals, netF* matVals) {
	int row;
	for (row = 0; row < leftHeight; row++) {
		int col;
		for (col = 0; col < rightWidth; col++) {
			netF sum = 0;
			int idx;
			for (idx = 0; idx < leftWidth; idx++) {
				sum += leftVals[row * leftWidth + idx] * rightVals[rightWidth * idx + col];
			}
			matVals[col + row * rightWidth] = sum;
		}
	}
}

matrix* multiplyMatrices(matrix* left, matrix* right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;

	if (leftWidth != rightHeight) {
		return NULL;
	}

	matrix* mat = createOrUseSuppliedMatrix(result, leftHeight, rightWidth);

	innerMultiplyMatrices(leftHeight, leftWidth, rightWidth, left->vals, right->vals, mat->vals);
	return mat;
}

matrix* elementMultMatrices(matrix* left, matrix* right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if ((leftWidth != rightWidth) || (leftHeight != rightHeight)) {
		return NULL;
	}
	matrix* mat = createOrUseSuppliedMatrix(result, leftHeight, leftWidth);
	netF* vals = mat->vals;
	int idx;
	for (idx = 0; idx < leftWidth * leftHeight; idx++) {
		mat->vals[idx] = left->vals[idx] * right->vals[idx];
	}
	return mat;
}

matrix* expandMultMatrices(matrix* left, matrix* right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if (leftHeight != rightHeight) {
		return NULL;
	}
	int totalWidth = leftWidth * rightWidth;
	matrix* mat = createOrUseSuppliedMatrix(result, leftHeight, totalWidth);
	netF* vals = mat->vals;
	int row;
	for (row = 0; row < leftHeight; row++) {
		int leftCol;
		for (leftCol = 0; leftCol < leftWidth; leftCol++) {
			int rightCol;
			for (rightCol = 0; rightCol < rightWidth; rightCol++) {
				mat->vals[row * totalWidth + leftCol * rightWidth + rightCol] =
					left->vals[row * leftWidth + leftCol] * right->vals[row * rightWidth + rightCol];
			}
		}
	}
	return mat;
}

static __inline netF dotProduct(netF* left, netF* right, const int count) {
	netF sum = 0;
	int idx = 0;
	for (idx = 0; idx < count; idx++) {
		sum += left[idx] * right[idx];
	}
	return sum;
}

/*static __inline netF alignedFastDotProduct(netF* left, netF* right, const int rounds, const int rem) {
	int n;
	int m = 0;
	__m128d mSum = _mm_set_pd(0, 0);
	__m128d* mLeft = (__m128d*)left;
	__m128d* mRight = (__m128d*)right;
	for (n = rounds; n--;) {
		__m128d a = mLeft[m];
		__m128d b = mRight[m];
		m++;
		__m128d dotOne = _mm_dp_pd(a, b, 0xff);
		__m128d e = mLeft[m];
		__m128d f = mRight[m];
		m++;
		__m128d dotTwo = _mm_dp_pd(e, f, 0xff);
		mSum = _mm_add_sd(dotOne, mSum);
		mSum = _mm_add_sd(dotTwo, mSum);
	}
	netF sum = _mm_cvtsd_f64(mSum);
	int idx = 4 * rounds;
	switch (rem)
	{
	case 3:
		sum += left[idx] * right[idx];
		idx++;
	case 2:
		sum += left[idx] * right[idx];
		idx++;
	case 1:
		sum += left[idx] * right[idx];
		idx++;
	}
	return sum;
}*/

static __inline netF fastDotProduct(netF* left, netF* right, const int rounds, const int rem) {
	int n;
	int idx = 0;
	netF sum = 0;
	for (n = rounds; n--;) {
		sum += left[idx] * right[idx];
		idx++;
		sum += left[idx] * right[idx];
		idx++;
		sum += left[idx] * right[idx];
		idx++;
		sum += left[idx] * right[idx];
		idx++;
	}
	switch (rem)
	{
	case 3:
		sum += left[idx] * right[idx];
		idx++;
		/* no break */
	case 2:
		sum += left[idx] * right[idx];
		idx++;
		/* no break */
	case 1:
		sum += left[idx] * right[idx];
		idx++;
	}
	return sum;
}

static __inline void innerCLTransExpandMultiplyMatrices(const int leftHeight, const int rightHeight, const int leftWidth,
	netF* leftVals, netF* rightVals, netF* matVals) {

	int totalWidth = leftHeight * rightHeight;
	size_t leftSize = sizeof(float) *  leftWidth * leftHeight;
	size_t rightSize = sizeof(float) * leftWidth * rightHeight;
	size_t matSize = sizeof(float) * totalWidth;

	cl_int err;
	err = clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputA, CL_TRUE, 0, leftSize, leftVals, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputB, CL_TRUE, 0, rightSize, rightVals, 0, NULL, NULL);

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
	err = clEnqueueNDRangeKernel(globalClSettings.queue, kernel, 1, NULL, globalSizes, NULL, 0, NULL, NULL);
	if (err) {
		printf("Error %d When executing the kernel.\n", err);
	}

	// Wait for the command queue to get serviced before reading back results
	clFinish(globalClSettings.queue);

	clEnqueueReadBuffer(globalClSettings.queue, globalClKernels.outputC, CL_TRUE, 0, matSize, matVals, 0, NULL, NULL);

	int col;
	for (col = 0; col < totalWidth; col++) {
		matVals[col] /= leftWidth;
	}
}

matrix* gpuTransExpandMultCollapseMatrices(matrix* left, matrix *right, matrix* result) {
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

matrix* expandMultCollapseMatrices(matrix* left, matrix *right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if (leftHeight != rightHeight) {
		return NULL;
	}
	int totalWidth = leftWidth * rightWidth;
	matrix* mat = createOrUseSuppliedMatrix(result, 1, totalWidth);
	fillMatrixZero(mat);
	netF* vals = mat->vals;
	int row;
	for (row = 0; row < leftHeight; row++) {
		int leftCol;
		for (leftCol = 0; leftCol < leftWidth; leftCol++) {
			netF leftVal = left->vals[row * leftWidth + leftCol];
			int rightCol;
			for (rightCol = 0; rightCol < rightWidth; rightCol++) {
				mat->vals[leftCol * rightWidth + rightCol] += leftVal * right->vals[row * rightWidth + rightCol];
			}
		}
	}
	int col;
	for (col = 0; col < totalWidth; col++) {
		mat->vals[col] /= leftHeight;
	}
	return mat;
}


static __inline void innerTransMultiplyMatrices(const int leftHeight, const int leftWidth, const int rightHeight, netF* leftVals, netF* rightVals, netF* matVals) {
	const int rounds = leftWidth >> 2;
	const int rem = leftWidth & 0x3;
	int row;
	int col;
	//#pragma omp parallel for default(shared) private(row, col) schedule(static)
	for (row = 0; row < leftHeight; row++) {
		for (col = 0; col < rightHeight; col++) {
			matVals[row * rightHeight + col] = fastDotProduct(leftVals + row * leftWidth, rightVals + leftWidth * col, rounds, rem);
		}
	}
}

static __inline void innerCLTransMultiplyMatrices(const int leftHeight, const int leftWidth, const int rightHeight, netF* leftVals, netF* rightVals, netF* matVals) {
	size_t totalItems = leftHeight * rightHeight; 
	size_t leftSize = sizeof(float) *  leftWidth * leftHeight;
	size_t rightSize = sizeof(float) * leftWidth * rightHeight;
	size_t matSize = sizeof(float) * totalItems;

	// Write our data set into the input array in device memory
	cl_int err;
	err = clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputA, CL_TRUE, 0, leftSize, leftVals, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(globalClSettings.queue, globalClKernels.inputB, CL_TRUE, 0, rightSize, rightVals, 0, NULL, NULL);
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
	clEnqueueReadBuffer(globalClSettings.queue, globalClKernels.outputC, CL_TRUE, 0, matSize, matVals, 0, NULL, NULL);
}

#define OUTER_TRANS_MULT_MATRIX(left, right, result) \
int leftWidth = left->width; \
int leftHeight = left->height; \
int rightWidth = right->width; \
int rightHeight = right->height; \
if (leftWidth != rightWidth) { \
	return NULL; \
} \
matrix* mat = createOrUseSuppliedMatrix(result, leftHeight, rightHeight); \


matrix* transMultiplyMatrices(matrix* left, matrix* right, matrix* result) {
	OUTER_TRANS_MULT_MATRIX(left, right, result)
	int useCpu = 1;
	if (useCpu) {
		innerTransMultiplyMatrices(leftHeight, leftWidth, rightHeight, left->vals, right->vals, mat->vals);
	} else {
		innerCLTransMultiplyMatrices(leftHeight, leftWidth, rightHeight, left->vals, right->vals, mat->vals);
	}
	return mat;
}


matrix* cpuTransMultiplyMatrices(matrix* left, matrix* right, matrix* result) {
	OUTER_TRANS_MULT_MATRIX(left, right, result)
	innerTransMultiplyMatrices(leftHeight, leftWidth, rightHeight, left->vals, right->vals, mat->vals);
	return mat;
}

matrix* gpuTransMultiplyMatrices(matrix* left, matrix* right, matrix* result) {
	OUTER_TRANS_MULT_MATRIX(left, right, result)
	innerCLTransMultiplyMatrices(leftHeight, leftWidth, rightHeight, left->vals, right->vals, mat->vals);
	return mat;
}

matrix* subtractMatrices(matrix* left, matrix* right, matrix* result) {
	if ((left->width != right->width) || (left->height != right->height)) {
		return NULL;
	}
	matrix* mat = createOrUseSuppliedMatrix(result, left->height, left->width);
	int row;
	for (row = 0; row < mat->height; row++) {
		int col;
		for (col = 0; col < mat->width;col++) {
			int idx = row * mat->width + col;
			mat->vals[idx] = left->vals[idx] - right->vals[idx];
		}
	}
	return mat;
}

matrix* addMatrices(matrix* left, matrix* right, matrix* result) {
	if ((left->width != right->width) || (left->height != right->height)) {
		return NULL;
	}
	matrix* mat = createOrUseSuppliedMatrix(result, left->height, left->width);
	int row;
	for (row = 0; row < mat->height; row++) {
		int col;
		for (col = 0; col < mat->width;col++) {
			int idx = row * mat->width + col;
			mat->vals[idx] = left->vals[idx] + right->vals[idx];
		}
	}
	return mat;
}

static __inline void innerTransposeMatrix(netF* original, netF* result, int originalHigh, int originalWide) {
	int row;
	for (row = 0; row < originalHigh; row++) {
		int col;
		for (col = 0; col < originalWide; col++) {
			result[col * originalHigh + row] = original[row * originalWide + col];
		}
	}
}

matrix* transposeMatrix(matrix* original, matrix* result) {
	matrix* mat = createOrUseSuppliedMatrix(result, original->width, original->height);
	innerTransposeMatrix(original->vals, result->vals, original->height, original->width);
	return mat;
}

matrix* transposeMatrixSelf(matrix* original, netF* extraData) {
	innerTransposeMatrix(original->vals, extraData, original->height, original->width);
	setMatrixValues(original, extraData);
	int oldHeight = original->height;
	original->height = original->width;
	original->width = oldHeight;
	return original;
}

netF sumSquareMatrix(matrix* mat) {
	netF total = 0;
	int idx;
	for (idx = 0; idx < (mat->width * mat->height); idx++) {
		total += mat->vals[idx] * mat->vals[idx];
	}
	return total;
}

matrix* cloneMatrix(matrix* original, matrix* result) {
	matrix* mat = createOrUseSuppliedMatrix(result, original->height, original->width);
	return setMatrixValues(mat, original->vals);
}

matrix* agumentMatrix(matrix* left, matrix* right, matrix* result) {
	if (left->height != right->height) {
		return NULL;
	}
	matrix* mat = createOrUseSuppliedMatrix(result, left->height, left->width + right->width);
	int yn;
	for (yn = 0; yn < left->height; yn++) {
		int xn;
		for (xn = 0; xn < left->width; xn++) {
			setMatrixVal(mat, yn, xn, *getMatrixVal(left, yn, xn));
		}
		for (xn = 0; xn < right->width; xn++) {
			setMatrixVal(mat, yn, left->width + xn, *getMatrixVal(right, yn, xn));
		}
	}
	return mat;
}

int doesMatrixHaveNANs(matrix* mat) {
	if (mat == NULL) {
		return 1;
	}
	int i;
	for (i = 0; i < mat->width * mat->height; i++) {
		if (isnan(mat->vals[i])) {
			return 1;
		}
	}
	return 0;
}