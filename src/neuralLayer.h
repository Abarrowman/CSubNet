#ifndef NEURALLAYER_H_
#define NEURALLAYER_H_

#include "matrix.h"
#include "coreDefs.h"

struct neuralLayerStub;

typedef struct neuralLayerStub {
	matrix* matrix;
	netF* biases;
} neuralLayer;

matrix* applyLayer(neuralLayer* layer, matrix* input, matrix* output);

neuralLayer* cloneLayer(neuralLayer* original, neuralLayer* result);

neuralLayer* createLayer(int inputs, int outputs);

neuralLayer* createLayerWithValues(int inputs, int outputs, netF* coefficients, netF* biases);

void deleteLayer(neuralLayer* layer);

int getLayerInputs(neuralLayer* layer);
int getLayerOutputs(neuralLayer* layer);


#endif
