#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include "coreDefs.h"

struct matrixStub;

typedef struct matrixStub {
	netF* vals;
	int width;
	int height;
} matrix;

//constructors
matrix* initMatrix(matrix* matrix, int height, int width);
matrix* createMatrix(int height, int width);
matrix* createIdentityMatrix(int height, int width);
matrix* createMatrixWithValues(int height, int width, netF* vals);
matrix* cloneMatrix(matrix* original, matrix* result);
matrix* agumentMatrix(matrix* left, matrix* right, matrix* result);


//deconstructor
void deleteMatrix(matrix* mat);
void clearMatrix(matrix* mat);

//mutators
matrix* fillMatrixZero(matrix* m);
matrix* fillMatrixIdentity(matrix* m);
matrix* setMatrixVal(matrix* mat, int row, int col, netF val);
netF* getMatrixVal(matrix* mat, int row, int col);
matrix* setMatrixValues(matrix *mat, netF *dubs);

//algebra
matrix* multiplyMatrices(matrix* left, matrix *right, matrix* result);
matrix* transMultiplyMatrices(matrix* left, matrix *right, matrix* result);
matrix* subtractMatrices(matrix* left, matrix* right, matrix* result);
matrix* addMatrices(matrix* left, matrix* right, matrix* result);
netF sumSquareMatrix(matrix* mat);


//misc
matrix* writeMatrix(FILE *stream, matrix* mat, char colSep, char rowSep);
void writeMatrixRow(FILE *stream, matrix* mat, char colSep, char rowSep, int row);
matrix* printMatrix(matrix* mat);



#endif
