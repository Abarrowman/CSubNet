#ifndef _CSV_H
#define _CSV_H

#include <stdio.h>
#include "utils.h"
#include "matrix.h"

struct csvStub;

typedef struct csvStub {
	int rows;
	int cols;
	stringFragment* contents;
	char* csvText;
} csv;

csv* readCSV(char* file);
void deleteCSV(csv* data);
stringFragment* getCSVCel(csv* data, int row, int col);
stringFragment* getCSVHeader(csv* data, int col);

matrix* extractMatrixFromCSV(csv* data, int row, int col, int wide, int high);
matrix* readMatrixFromCSV(char* file);
matrix* writeMatrixAsCSVToFile(char* file, matrix* mat, csv* header, int labels, int inputs, int outputs);



#endif
