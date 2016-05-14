#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "coreDefs.h"
#include "neuralLayer.h"
#include "matrix.h"
#include "utils.h"

__inline netF layerSigmoid(netF input) {
		//return 1 / (1 + exp(-input));
		return (input / (1 + fabsf(input)) + 1) / 2;
}
	
static __inline void applyBiases(netF* __restrict vals, netF* __restrict biases, const int outHigh, const int outWide) {
	int row;
	int col;
	int idx;
	#pragma omp parallel for default(shared) private(row, col, idx) schedule(static)
	for (row = 0; row < outHigh; row++) {
		for (col = 0; col < outWide; col++) {
			idx = col + row * outWide;
			vals[idx] = layerSigmoid(vals[idx] - biases[col]);
		}
	}
}

matrix* applyLayer(neuralLayer* layer, matrix* input, matrix* output) {
	//pre trans
	//multiplyMatrices(input, layer->matrix, output);

	//post trans
	transMultiplyMatrices(input, layer->matrix, output);

	if (output == NULL) {
		return NULL;
	}

	applyBiases(output->vals, layer->biases, output->height, output->width);
	return output;
}

neuralLayer* cloneLayer(neuralLayer* original, neuralLayer* result) {
	neuralLayer* layer;
	//pre trans
	/*int outputs = original->matrix->width;
	if (result == NULL || (result->matrix->width != outputs)
			|| (result->matrix->height != original->matrix->height)) {
		layer = createLayer(original->matrix->height, outputs);
	} else {
		layer = result;
	}*/

	//post trans
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

	//pre trans
	//layer->matrix = createIdentityMatrix(inputs, outputs);

	//post trans
	layer->matrix = createIdentityMatrix(outputs, inputs);

	layer->biases = (netF*)calloc(outputs, sizeof(netF)); //unbaised
	return layer;
}

neuralLayer* createLayerWithValues(int inputs, int outputs, netF* coefficients, netF* biases) {
	neuralLayer* layer = (neuralLayer*)malloc(sizeof(neuralLayer));

	//pre trans
	//layer->matrix = createMatrixWithValues(inputs, outputs, coefficients);

	//post trans
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
	//pre trans
	//return layer->matrix->height;

	//post trans
	return layer->matrix->width;
}

int getLayerOutputs(neuralLayer* layer) {
	//pre trans
	//return layer->matrix->width;

	//post trans
	return layer->matrix->height;
}
