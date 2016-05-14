#ifndef TRAININGDATA_H_
#define TRAININGDATA_H_

#include "matrix.h"

struct trainingDataStub;

typedef struct trainingDataStub {
	matrix* input;
	matrix* output;
} trainingData;

void clearTrainingData(trainingData* datas);
trainingData* readTrainingData(trainingData* data, char* file, int inputCols, int labelCols);
int getTrainingDataIntputs(trainingData* datas);
int getTrainingDataOutputs(trainingData* datas);
int getTrainingDataCount(trainingData* datas);


#endif
