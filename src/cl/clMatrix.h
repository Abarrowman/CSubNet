#ifndef _CL_MATRIX_H
#define _CL_MATRIX_H

#include "../matrix.h"

matrix* gpuTransExpandMultCollapseMatrices(
	matrix* left, matrix *right, matrix* result);

matrix* gpuTransMultiplyMatrices(matrix* left, matrix *right, matrix* result);
#endif
