#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "coreDefs.h"
#include "neuralLayer.h"
#include "matrix.h"
#include "utils.h"

__inline netF layerSigmoid(netF input) {
	return (netF)((input / (1 + fabs(input)) + 1) / 2);
}

__inline netF layerSigmoidDerivative(netF input) {
	netF denom = (netF)(fabs(input) + 1);
	return 1 / (2 * denom * denom);
}

static __inline void applySigmoidDerivative(netF* __restrict vals, int count) {
	int idx;
	for (idx = 0; idx < count; idx++) {
		vals[idx] = layerSigmoidDerivative(vals[idx]);
	}
}

matrix* calculateLayerOutputDerrivative(neuralLayer* layer, matrix* preOutput) {
	applySigmoidDerivative(preOutput->vals, preOutput->width * preOutput->height);
	return preOutput;
}

static __inline void applyBiases(netF* __restrict vals, netF* __restrict biases, const int outHigh, const int outWide) {
	int row;
	int col;
	int idx;
	for (row = 0; row < outHigh; row++) {
		for (col = 0; col < outWide; col++) {
			idx = col + row * outWide;
			vals[idx] = vals[idx] - biases[col];
		}
	}
}

static __inline void applySigmoid(netF* __restrict vals, int count) {
	int idx;
	for (idx = 0; idx < count; idx++) {
		vals[idx] = layerSigmoid(vals[idx]);
	}
}

matrix* applyLayer(neuralLayer* layer, matrix* input, matrix* output, int shouldApplySigmoid, matrix* preOutput) {
	cpuTransMultiplyMatrices(input, layer->matrix, output);

	if (output == NULL) {
		return NULL;
	}		
	applyBiases(output->vals, layer->biases, output->height, output->width);
	if (preOutput) {
		cloneMatrix(output, preOutput);
	}
	if (shouldApplySigmoid) {
		applySigmoid(output->vals, output->height * output->width);
	}
	return output;
}

neuralLayer* cloneLayer(neuralLayer* original, neuralLayer* result) {
	neuralLayer* layer;
	int inputs = original->matrix->width;
	int outputs = original->matrix->height;
	if (result == NULL || (result->matrix->height != outputs)
		|| (result->matrix->width != inputs)) {
		layer = createLayer(inputs, outputs);
	} else {
		layer = result;
	}

	cloneMatrix(original->matrix, layer->matrix);
	memcpy(layer->biases, original->biases, sizeof(netF) * outputs);
	return layer;
}

neuralLayer* createLayer(int inputs, int outputs) {
	neuralLayer* layer = (neuralLayer*)malloc(sizeof(neuralLayer));
	layer->matrix = createIdentityMatrix(outputs, inputs);
	fillMatrixRandom(layer->matrix); //randomize matrix
	layer->biases = (netF*)calloc(outputs, sizeof(netF)); //unbaised
	return layer;
}

neuralLayer* createLayerWithValues(int inputs, int outputs, netF* coefficients, netF* biases) {
	neuralLayer* layer = (neuralLayer*)malloc(sizeof(neuralLayer));
	layer->matrix = createMatrixWithValues(outputs, inputs, coefficients);

	layer->biases = (netF*)malloc(sizeof(netF) * outputs);
	memcpy(layer->biases, biases, sizeof(netF) * outputs);
	return layer;
}

void deleteLayer(neuralLayer* layer) {
	deleteMatrix(layer->matrix);
	layer->matrix = NULL;
	free(layer->biases);
	layer->biases = NULL;
	free(layer);
}

int getLayerInputs(neuralLayer* layer) {
	return layer->matrix->width;
}

int getLayerOutputs(neuralLayer* layer) {
	return layer->matrix->height;
}
