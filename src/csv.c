#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include "csv.h"
#include "utils.h"
#include "matrix.h"
#include "coreDefs.h"

#define MAX_CSV_SIZE (1024 * 1024 * 10)
#define CSV_INCLUDE_DEBUG_LOGS 0
#define CSV_INCLUDE_ERROR_LOGS 1



void deleteCSV(csv* data) {
	if (data->csvText == NULL) {
		return;
	}
	free(data->csvText);
	data->csvText = NULL;
	free(data->contents);
	data->contents = NULL;
	free(data);
}

static matrix* writeMatrixCSV(FILE* stream, matrix* mat, csv* header, int labels, int inputs, int outputs) {
	int col;
	int headerCols = 0;
	if (header) {
		headerCols = header->cols;
		if (headerCols < labels) {
			return NULL;
		}
	} else {
		if (labels != 0) {
			return NULL;
		}
	}
	int cols = inputs + outputs + labels;
	int lastCol = cols - 1;
	int inputAndLabelCols = inputs + labels;


	stringFragment* frag;
	for (col = 0; col < lastCol; col++) {
		if (headerCols > col) {
			frag = getCSVHeader(header, col);
			fprintf(stream, "\"%.*s\",", (int)frag->length, frag->start);
		} else if (col < inputAndLabelCols) {
			fprintf(stream, "Input_%d,", col - labels);
		} else {
			fprintf(stream, "Output_%d,", col - inputAndLabelCols);
		}
	}
	if (header->cols > col) {
		frag = getCSVHeader(header, col);
		fprintf(stream, "\"%.*s\"", (int)frag->length, frag->start);
	} else if (col < inputAndLabelCols) {
		fprintf(stream, "Input_%d", col - labels);
	} else {
		fprintf(stream, "Output_%d", col - inputAndLabelCols);
	}
	fprintf(stream, "\n");
	char colSep = ',';
	char rowSep = '\n';
	if (labels == 0) {
		return writeMatrix(stream, mat, colSep, rowSep);
	} else {
		int row;
		for (row = 0; row < mat->height; row++) {
			int labelIdx ;
			for (labelIdx = 0; labelIdx < labels; labelIdx++) {
				frag = getCSVCel(header, row, labelIdx);
				fprintf(stream, "\"%.*s\",", (int)frag->length, frag->start);
			}
			writeMatrixRow(stream, mat, colSep, rowSep, row);
		}
		return mat;
	}
}

matrix* writeMatrixAsCSVToFile(char* file, matrix* mat, csv* header, int labels, int inputs, int outputs) {

	FILE* fp;
	errno_t err = fopen_s(&fp, file, "wb");
	if (err != 0) {
		return NULL;
	}
	mat = writeMatrixCSV(fp, mat,header, labels, inputs, outputs);
	if (fclose(fp)) {
		return NULL;
	}
	return mat;
}

csv* readCSV(char* file) {
	stringFragment csvFile = readFileIntoMemory(file, MAX_CSV_SIZE);
	char* csvText = csvFile.start;
	size_t csvLen = csvFile.length;
	if (csvText == NULL) {
		PRINT_FLUSH(CSV_INCLUDE_ERROR_LOGS, "An error occured while opening the file [%s].\n", file);
		return NULL;
	}

	int numLines = 0;
	int numCols = 1;
	size_t idx = 0;
	char c;
	int newLineOk = 0;
	int quoted = 0;
	for (idx = 0; idx < csvLen; idx++) {
		c = csvText[idx];
		if (quoted) {
			if (c == '"') {
				quoted = 0;
			}
		} else {
			if (c == '"') {
				quoted = 1;
			} else if ((c == '\n') || (c == '\r')) {
				if (newLineOk) {
					numLines++;
				}
				newLineOk = 0;
			} else if (c != ' ') {
				newLineOk = 1;
				if ((numLines == 0) && (c == ',')) {
					numCols++;
				}
			}
		}
	}

	PRINT_FLUSH(CSV_INCLUDE_DEBUG_LOGS, "Lines: %d, Columns: %d \n", numLines, numCols);

	int rows = numLines - 1;

	csv* data = malloc(sizeof(csv));
	data->contents = malloc(sizeof(stringFragment) * numCols * numLines);
	data->csvText = csvText;
	data->rows = rows;
	data->cols = numCols;

	quoted = 0;
	idx = 0;
	char* start = csvText;
	int length = 0;
	newLineOk = 0;
	int colIdx = 0;
	int rowIdx = 0;
	stringFragment* frag;
	for (idx = 0; idx < csvLen; idx++) {
		c = csvText[idx];
		if (quoted) {
			if (c == '"')  {
				quoted = 0;
				length--;
			}
		} else {
			if ((c == '\n') || (c == '\r')) {
				if (newLineOk) {
					frag = data->contents + colIdx + numCols * rowIdx;
					frag->start = start;
					frag->length = length;
					PRINT_FLUSH(CSV_INCLUDE_DEBUG_LOGS, "Frag (%d,%d) [%.*s] \n", rowIdx, colIdx, length, start);
					rowIdx++;
					colIdx = 0;
				}
				newLineOk = 0;
			} else if (c != ' ') {
				if (!newLineOk) {
					newLineOk = 1;
					start = csvText + idx;
					length = 0;

				}
				if (c == '"') {
					start++;
					length = -1;
					quoted = 1;
				} else if (c == ',') {
					frag = data->contents + colIdx + numCols * rowIdx;
					frag->start = start;
					frag->length = length;
					PRINT_FLUSH(CSV_INCLUDE_DEBUG_LOGS, "Frag (%d,%d) [%.*s] \n", rowIdx, colIdx, length, start);
					colIdx++;
					start = csvText + idx + 1; //starts after the comma
					length = -1;
				}
			}
		}
		length++;
	}

	return data;
}

stringFragment* getCSVCel(csv* data, int row, int col) {
	return data->contents + col + (row + 1) * data->cols;
}

netF getCSVCelAsNetF(csv* data, int row, int col) {
	return parseNetF(getCSVCel(data, row, col));
}

stringFragment* getCSVHeader(csv* data, int col) {
	return getCSVCel(data, -1, col);
}

matrix* extractMatrixFromCSV(csv* data, int topRow, int leftCol, int wide, int high) {
	if (data->rows < (high + topRow)) {
		PRINT_FLUSH(CSV_INCLUDE_ERROR_LOGS, "CSV has %d rows thus it can not extract row at index %d.\n", data->rows, high + topRow - 1);
		return NULL;
	}
	if (data->cols < (wide + leftCol)) {
		PRINT_FLUSH(CSV_INCLUDE_ERROR_LOGS, "CSV has %d columns thus it can not extract column at index %d.\n", data->cols, wide + leftCol - 1);
		return NULL;
	}
	matrix* mat = createMatrix(high, wide);
	if (!mat) {
		PRINT_FLUSH(CSV_INCLUDE_ERROR_LOGS, "An error occured while allocating Matrix for CSV.\n");
		return NULL;
	}

	int yn;
	for (yn = 0; yn < high; yn++) {
		int xn;
		for (xn = 0; xn < wide; xn++) {
			netF val = getCSVCelAsNetF(data, yn + topRow, xn + leftCol);
			if (isnan(val)){
				deleteMatrix(mat);
				stringFragment* frag = getCSVCel(data, yn + topRow, xn + leftCol);
				PRINT_FLUSH(CSV_INCLUDE_ERROR_LOGS, "The value at row index %d column index %d has a value [%.*s] which is not a number.", yn + topRow, xn + leftCol, (int)frag->length, frag->start);
				return NULL;
			} else {
				setMatrixVal(mat, yn, xn, val);
			}
		}
	}
	return mat;
}

matrix* readMatrixFromCSV(char* file) {
	csv* data = readCSV(file);
	if (!data) {
		return NULL;
	}
	matrix* mat = extractMatrixFromCSV(data, 0, 0, data->cols, data->rows);
	deleteCSV(data);
	return mat;
}
