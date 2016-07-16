#include <stdlib.h>
#include "utils.h"
#include "trainingData.h"
#include "matrix.h"
#include "csv.h"

void clearTrainingData(trainingData* data) {
	if (data->input == NULL) {
		return;
	}
	deleteMatrix(data->input);
	deleteMatrix(data->output);
	data->input = NULL;
	data->output = NULL;
}

trainingData* readTrainingData(trainingData* data, char* file, int inputCols, int labelCols) {
	csv* csvData =  readCSV(file);
	if (csvData == NULL) {
		return NULL;
	}

	data->input = extractMatrixFromCSV(csvData, 0, labelCols, inputCols, csvData->rows);
	data->output = extractMatrixFromCSV(csvData, 0, inputCols + labelCols,
			csvData->cols - inputCols - labelCols, csvData->rows);

	deleteCSV(csvData);
	if (!data->input || !data->output) {
		return NULL;
	}
	return data;
}

int getTrainingDataIntputs(trainingData* datas) {
	return datas->input->width;
}
int getTrainingDataOutputs(trainingData* datas) {
	return datas->output->width;
}
int getTrainingDataCount(trainingData* datas) {
	return datas->input->height;
}
