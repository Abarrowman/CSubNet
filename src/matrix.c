#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix.h"
#include "utils.h"

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

void validateSuppliedMatrix(matrix* supplied, int height, int width) {
	if (supplied == NULL) {
		PRINT_FLUSH(1, "Supplied matrix is NULL.");
		exit(1);
	} else if ((supplied->width != width) || (supplied->height != height)) {
		PRINT_FLUSH(1, "Supplied matrix is of the wrong size %dx%d instead of %dx%d",
			supplied->height, supplied->width, height, width);
		exit(1);
	}
}

static __inline void innerMultiplyMatrices(const int leftHeight, const int leftWidth, const int rightWidth,
	const netF* leftVals, const netF* rightVals, netF* matVals) {
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

	validateSuppliedMatrix(result, leftHeight, rightWidth);

	innerMultiplyMatrices(leftHeight, leftWidth, rightWidth, left->vals, right->vals, result->vals);
	return result;
}

matrix* elementMultMatrices(matrix* left, matrix* right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if ((leftWidth != rightWidth) || (leftHeight != rightHeight)) {
		return NULL;
	}
	validateSuppliedMatrix(result, leftHeight, leftWidth);
	int idx;
	for (idx = 0; idx < leftWidth * leftHeight; idx++) {
		result->vals[idx] = left->vals[idx] * right->vals[idx];
	}
	return result;
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
	validateSuppliedMatrix(result, leftHeight, totalWidth);
	int row;
	for (row = 0; row < leftHeight; row++) {
		int leftCol;
		for (leftCol = 0; leftCol < leftWidth; leftCol++) {
			int rightCol;
			for (rightCol = 0; rightCol < rightWidth; rightCol++) {
				result->vals[row * totalWidth + leftCol * rightWidth + rightCol] =
					left->vals[row * leftWidth + leftCol] * right->vals[row * rightWidth + rightCol];
			}
		}
	}
	return result;
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
	validateSuppliedMatrix(result, 1, totalWidth);
	fillMatrixZero(result);
	int row;
	for (row = 0; row < leftHeight; row++) {
		int leftCol;
		for (leftCol = 0; leftCol < leftWidth; leftCol++) {
			netF leftVal = left->vals[row * leftWidth + leftCol];
			int rightCol;
			for (rightCol = 0; rightCol < rightWidth; rightCol++) {
				result->vals[leftCol * rightWidth + rightCol] += leftVal * right->vals[row * rightWidth + rightCol];
			}
		}
	}
	int col;
	for (col = 0; col < totalWidth; col++) {
		result->vals[col] /= leftHeight;
	}
	return result;
}

static __inline netF dotProduct(const netF* left, const netF* right, const int count) {
	netF sum = 0;
	int idx = 0;
	for (idx = 0; idx < count; idx++) {
		sum += left[idx] * right[idx];
	}
	return sum;
}

static __inline void innerTransExpandMultCollapseMatrices(const int leftHeight, const int rightHeight,
		const int leftWidth, const netF* leftVals, const netF* rightVals, netF* result) {
	int leftRow;
	#pragma omp parallel for schedule(dynamic, 16)
	for (leftRow = 0; leftRow < leftHeight; leftRow++) {
		int rightRow = 0;
		for (rightRow = 0; rightRow < rightHeight; rightRow++) {
			netF sum = 0;
			result[leftRow * rightHeight + rightRow] = dotProduct(
				leftVals + leftRow * leftWidth, rightVals + rightRow * leftWidth, leftWidth);
		}
	}
}

matrix* transExpandMultCollapseMatrices(matrix* left, matrix *right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if (leftWidth != rightWidth) {
		return NULL;
	}
	int totalHeight = leftHeight * rightHeight;
	validateSuppliedMatrix(result, 1, totalHeight);

	innerTransExpandMultCollapseMatrices(leftHeight, rightHeight,
		leftWidth, left->vals, right->vals, result->vals);

	int col;
	for (col = 0; col < totalHeight; col++) {
		result->vals[col] /= leftWidth;
	}
	return result;
}

static __inline void innerTransMultiplyMatrices(const int leftHeight, const int leftWidth, const int rightHeight,
		const netF* leftVals, const netF* rightVals, netF* matVals) {
	int row;
	#pragma omp parallel for schedule(dynamic, 16)
	for (row = 0; row < leftHeight; row++) {
		for (int col = 0; col < rightHeight; col++) {
			matVals[row * rightHeight + col] = dotProduct(
				leftVals + row * leftWidth, rightVals + leftWidth * col, leftWidth);
		}
	}
}

matrix* cpuTransMultiplyMatrices(matrix* left, matrix* right, matrix* result) {
	int leftWidth = left->width;
	int leftHeight = left->height;
	int rightWidth = right->width;
	int rightHeight = right->height;
	if (leftWidth != rightWidth) {
		return NULL;
	}
	validateSuppliedMatrix(result, leftHeight, rightHeight);
	innerTransMultiplyMatrices(leftHeight, leftWidth, rightHeight, left->vals, right->vals, result->vals);
	return result;
}

matrix* subtractMatrices(matrix* left, matrix* right, matrix* result) {
	if ((left->width != right->width) || (left->height != right->height)) {
		return NULL;
	}
	validateSuppliedMatrix(result, left->height, left->width);
	int row;
	for (row = 0; row < result->height; row++) {
		int col;
		for (col = 0; col < result->width;col++) {
			int idx = row * result->width + col;
			result->vals[idx] = left->vals[idx] - right->vals[idx];
		}
	}
	return result;
}

matrix* addMatrices(matrix* left, matrix* right, matrix* result) {
	if ((left->width != right->width) || (left->height != right->height)) {
		return NULL;
	}
	validateSuppliedMatrix(result, left->height, left->width);
	int row;
	for (row = 0; row < result->height; row++) {
		int col;
		for (col = 0; col < result->width;col++) {
			int idx = row * result->width + col;
			result->vals[idx] = left->vals[idx] + right->vals[idx];
		}
	}
	return result;
}

static __inline void innerTransposeMatrix(const netF* original, netF* result, const int origHigh,
		const int origWide) {
	/*
	Transpose blocks of the matrix at a time which is slightly cache friendlier for large matrices.
	*/
	const int block = 32;
	int rowBlock;
	for (rowBlock = 0; rowBlock < origHigh; rowBlock += block) {
		for (int colBlock = 0; colBlock < origWide; colBlock += block) {
			for (int row = rowBlock; row < rowBlock + block && row < origHigh; row++) {
				for (int col = colBlock; col < colBlock + block && col < origWide; col++) {
					result[col * origHigh + row] = original[row * origWide + col];
				}
			}
		}
	}
}

matrix* transposeMatrix(matrix* original, matrix* result) {
	validateSuppliedMatrix(result, original->width, original->height);
	innerTransposeMatrix(original->vals, result->vals, original->height, original->width);
	return result;
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
	validateSuppliedMatrix(result, original->height, original->width);
	return setMatrixValues(result, original->vals);
}

matrix* agumentMatrix(matrix* left, matrix* right, matrix* result) {
	if (left->height != right->height) {
		return NULL;
	}
	validateSuppliedMatrix(result, left->height, left->width + right->width);
	int yn;
	for (yn = 0; yn < left->height; yn++) {
		int xn;
		for (xn = 0; xn < left->width; xn++) {
			setMatrixVal(result, yn, xn, *getMatrixVal(left, yn, xn));
		}
		for (xn = 0; xn < right->width; xn++) {
			setMatrixVal(result, yn, left->width + xn, *getMatrixVal(right, yn, xn));
		}
	}
	return result;
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
