#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "neuralLayer.h"
#include "matrix.h"
#include "trainingData.h"
#include "utils.h"

struct neuralNetworkStub;

typedef struct neuralNetworkStub {
	neuralLayer** layers;
	int numLayers;
} neuralNetwork;

matrix* applyNetwork(neuralNetwork* network, matrix* input, matrix* output, matrix* intermediates);

neuralNetwork* swarmOptimizeNetwork(neuralNetwork* original, trainingData* train);
neuralNetwork* annealNetwork(neuralNetwork* original, trainingData* train);
neuralNetwork* evolveNetwork(neuralNetwork* original, trainingData* train);
neuralNetwork* gradientClimbNetwork(neuralNetwork* original, trainingData* train);

neuralNetwork* createEmptyNetwork(int numLayers);
neuralNetwork* createSizedNetwork(int* sizes, int numLayers);
neuralNetwork* createStandardNetwork(int inputSize, int outputSize);

void deleteNetwork(neuralNetwork* network);
neuralNetwork* cloneNeuralNetwork(neuralNetwork* network);
neuralNetwork* copyNeuralNetwork(neuralNetwork* network, neuralNetwork* target);

neuralNetwork* printNeuralNetwork(neuralNetwork* network);

int getNetworkInputs(neuralNetwork* network);
int getNetworkOutputs(neuralNetwork* network);

stringFragment encodeNeuralNetwork(neuralNetwork* network);
neuralNetwork* decodeNeuralNetwork(stringFragment frag);

neuralNetwork* readNeuralNetwork(char* file);


#endif
