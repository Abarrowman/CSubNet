#include "neuralLayer.h"
#include "neuralNetwork.h"
#include "trainingData.h"
#include "matrix.h"
#include "utils.h"
#include "coreDefs.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define NEURAL_NETWORK_INCLUDE_DEBUG_LOGS 1
#define MAX_NEURAL_NETWORK_SIZE (1024 * 1024 * 10)

static void deleteIntermediates(matrix* intermediates, neuralNetwork* network) {
	int idx;
	int numInters = network->numLayers - 1;
	for (idx = 0; idx < numInters; idx++) {
		clearMatrix(intermediates + idx);
	}
	free(intermediates);
}

static matrix* createNetworkIntermediates(neuralNetwork* network, int inputs) {
	int idx;
	int numInters = network->numLayers - 1;

 	matrix* intermediates = (matrix*)malloc(sizeof(matrix) * numInters);
	for (idx = 0; idx < numInters; idx++) {
		int size = getLayerOutputs(network->layers[idx]);
		initMatrix(intermediates + idx, inputs, size);
	}
	return intermediates;
}

matrix* applyNetwork(neuralNetwork* network, matrix* input, matrix* output, matrix* intermediates) {

	if (output == NULL) {
		output = createMatrix(input->height, getNetworkOutputs(network));
	}
	int needsIntermediates = intermediates == NULL;
	if (needsIntermediates) {
		intermediates = createNetworkIntermediates(network, input->height);
	}

	int idx;
	matrix* old = input;
	matrix* inter;
	for (idx = 0; idx < network->numLayers - 1; idx++) {
		inter = intermediates + idx;
		applyLayer(network->layers[idx], old, inter);
		old = inter;
	}
	applyLayer(network->layers[idx], old, output);
	if (needsIntermediates) {
		deleteIntermediates(intermediates, network);
	}
	return output;
}

static netF determineAcceptanceThreshold(netF temp, netF oldErr, netF newErr) {
	return expf((oldErr - newErr) * 500 / temp);
}

static netF calculateTrainingDataError(neuralNetwork* network, trainingData* trains, matrix* output, matrix* intermediates) {
	applyNetwork(network, trains->input, output, intermediates);
	subtractMatrices(output, trains->output, output);
	return sumSquareMatrix(output) / (output->height * output->width);
}

neuralNetwork* annealNetwork(neuralNetwork* original, trainingData* train) {

	neuralNetwork* current = cloneNeuralNetwork(original);
	neuralNetwork* best = cloneNeuralNetwork(original);
	neuralNetwork* mutant = cloneNeuralNetwork(original);
	matrix* output = createMatrix(train->output->height, getNetworkOutputs(original));

	matrix* intermediates = createNetworkIntermediates(original, getTrainingDataCount(train));


	int maxCycles = 500000;
	netF initialTemp = 5;
	netF temp = initialTemp;
	int cyclesWorse = 0;

	netF initialError = calculateTrainingDataError(original, train, output, intermediates);
	PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Initial %f\n", initialError);
	netF currentStateError = initialError;
	netF bestStateError = initialError;

	int cycle;
	for (cycle = 0; cycle < maxCycles; cycle++) {
		copyNeuralNetwork(current, mutant);

		int layerIdx;
		for (layerIdx = 0; layerIdx < mutant->numLayers; layerIdx++) {
			neuralLayer* layer = mutant->layers[layerIdx];
			matrix* mat = layer->matrix;
			int rowIdx;
			for (rowIdx = 0; rowIdx < mat->height; rowIdx++) {
				int colIdx;
				for (colIdx = 0; colIdx < mat->width; colIdx++) {
					netF* matVal = getMatrixVal(mat, rowIdx, colIdx);
					//*matVal += (randomNetF() - 0.5);
					*matVal += (randomNetF() - 0.5f) / 2;
				}
			}

			int biases = getLayerOutputs(layer);
			int biasIdx;
			for (biasIdx = 0; biasIdx < biases; biasIdx++) {
				layer->biases[biasIdx] += (randomNetF() - 0.5f);
			}
		}

		netF mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);
		if (cycle % 10000 == 0) {
			PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Cycle %d Temp %f Best %f Curr %f Mut %f\n", cycle, temp,
					bestStateError, currentStateError, mutantStateError);
			//printNeuralNetwork(current);
		}

		if (determineAcceptanceThreshold(temp, currentStateError, mutantStateError)
				> randomNetF()) {
			currentStateError = mutantStateError;
			copyNeuralNetwork(mutant, current);
		} else {
			cyclesWorse++;
			if (cyclesWorse == 1000) {
				cyclesWorse = 0;
				currentStateError = bestStateError;
				copyNeuralNetwork(best, current);
			}
		}

		if (mutantStateError < bestStateError) {
			bestStateError = mutantStateError;
			copyNeuralNetwork(mutant, best);
		}

		//temp = initialTemp - initialTemp * cycles / maxCycles;
		temp = initialTemp * expf(-3.5f * cycle / maxCycles);
	}

	deleteNetwork(current);
	deleteNetwork(mutant);
	deleteMatrix(output);
	deleteIntermediates(intermediates, original);

	PRINT_FLUSH(1, "Cycle %d Initial %f Best %f \n", cycle, initialError, bestStateError);

	return best;
}

neuralNetwork* createEmptyNetwork(int numLayers) {
	neuralNetwork* network = (neuralNetwork*)malloc(sizeof(neuralNetwork));
	network->layers = (neuralLayer**)calloc(numLayers, sizeof(neuralLayer*)); //zeroed memory contains only null pointers
	network->numLayers = numLayers;
	return network;
}

neuralNetwork* createSizedNetwork(int* sizes, int numLayers) {
	neuralNetwork* network = createEmptyNetwork(numLayers);
	int idx;
	int oldSize = sizes[0];
	for (idx = 0; idx < numLayers; idx++) {
		int newSize = sizes[idx + 1];
		network->layers[idx] = createLayer(oldSize, newSize);
		oldSize = newSize;
	}
	return network;
}

neuralNetwork* createStandardNetwork(int inputSize, int outputSize) {
	neuralNetwork* network = createEmptyNetwork(2);
	int middleSize = (inputSize + outputSize) / 2;
	network->layers[0] = createLayer(inputSize, middleSize);
	network->layers[1] = createLayer(middleSize, outputSize);
	return network;
}

void deleteNetwork(neuralNetwork* network) {
	if (network == NULL) {
		return;
	}
	int idx;
	for (idx = 0; idx < network->numLayers; idx++) {
		deleteLayer(network->layers[idx]);
	}
	free(network->layers);
	network->layers = NULL;
	free(network);
}

neuralNetwork* cloneNeuralNetwork(neuralNetwork* network) {
	neuralNetwork* clone = (neuralNetwork*)malloc(sizeof(neuralNetwork));
	clone->numLayers = network->numLayers;
	clone->layers = (neuralLayer**)malloc(sizeof(neuralLayer*) * clone->numLayers);
	int n;
	for (n = 0;n < clone->numLayers;n++) {
		clone->layers[n] = cloneLayer(network->layers[n], NULL);
	}
	return clone;
}

neuralNetwork* copyNeuralNetwork(neuralNetwork* network, neuralNetwork* target) {
	int n;
	for (n = 0;n < target->numLayers;n++) {
		cloneLayer(network->layers[n], target->layers[n]);
	}
	return target;
}

neuralNetwork* printNeuralNetwork(neuralNetwork* network) {
	int layerIdx;
	for (layerIdx = 0; layerIdx < network->numLayers; layerIdx++) {
		PRINT_FLUSH(1, "Layer %d\n", layerIdx);
		neuralLayer* layer = network->layers[layerIdx];
		printMatrix(layer->matrix);
		PRINT_FLUSH(1, "Biases ");
		int biasIdx;
		for (biasIdx = 0; biasIdx < getLayerOutputs(layer); biasIdx++) {
			PRINT_FLUSH(1, "%f ", layer->biases[biasIdx]);
		}
		PRINT_FLUSH(1, "\n");
	}
	return network;
}

int getNetworkOutputs(neuralNetwork* network) {
	return getLayerOutputs(network->layers[network->numLayers - 1]);
}

int getNetworkInputs(neuralNetwork* network) {
	return getLayerInputs(network->layers[0]);
}

stringFragment encodeNeuralNetwork(neuralNetwork* network) {
	size_t encodedSize = sizeof(int);//number of layers
	int layer = 0;
	for (layer = 0; layer < network->numLayers; layer++) {
		matrix* mat = network->layers[layer]->matrix;

		//pre trans
		//encodedSize += sizeof(netF) * mat->width * (mat->height + 1) + 2 * sizeof(int);

		//post trans
		encodedSize += sizeof(netF) * (mat->width + 1) * mat->height  + 2 * sizeof(int);
	}

	stringFragment frag = {
		.start = malloc(encodedSize),
		.length = encodedSize
	};

	if (frag.start == NULL) {
		return frag;
	}

	int* ints = (int*)frag.start;
	*ints = network->numLayers;
	ints++;

	for (layer = 0; layer < network->numLayers; layer++) {
		neuralLayer* neurLay = network->layers[layer];
		int inputs = getLayerInputs(neurLay);
		*ints = inputs;
		ints++;
		int outputs = getLayerOutputs(neurLay); 
		*ints = outputs;
		ints++;

		netF* netFs = (netF*)ints;
		int netFCount = inputs * outputs;
		memcpy(netFs, neurLay->matrix->vals, netFCount * sizeof(netF));
		netFs += netFCount;

		netFCount = outputs;
		memcpy(netFs, network->layers[layer]->biases, netFCount * sizeof(netF));
		ints = (int*)(netFs + netFCount);
	}
	//char* end = frag.start + encodedSize;
	return frag;
}

neuralNetwork* decodeNeuralNetwork(stringFragment frag) {
	int* ints = (int*)frag.start;
	int layers = *ints;
	ints++;
	neuralNetwork* network = createEmptyNetwork(layers);

	int layerIdx;
	for (layerIdx = 0; layerIdx < network->numLayers; layerIdx++) {
		int inputs = *ints;
		ints++;
		int outputs = *ints;
		ints++;
		netF* netFs = (netF*)ints;
		int coefficientCount = inputs * outputs;
		neuralLayer* layer = createLayerWithValues(inputs, outputs, netFs, netFs + coefficientCount);
		ints = (int*)(netFs + coefficientCount + outputs);
		network->layers[layerIdx] = layer;
	}
	return network;
}

neuralNetwork* readNeuralNetwork(char* file) {
	stringFragment frag = readFileIntoMemory(file, MAX_NEURAL_NETWORK_SIZE);
	if (frag.start == NULL) {
		return NULL;
	}
	neuralNetwork* network = decodeNeuralNetwork(frag);
	free(frag.start);
	return network;
}