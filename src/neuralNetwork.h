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

struct netOptSettingsStub;
typedef struct netOptSettingsStub {
	//stuck onto this structure for convenience but don't entirely belong here 
	int labelColumns;

	//general
	int maxRounds;
	int algorithm;

	//annealing
	int coolRounds;

	//swarm
	int particleCount;

	//evolution
	int childCount;

	//backprop
	int portionTrain;
} netOptSettings;

netOptSettings* initNetOptSettings(netOptSettings* settings);


matrix* applyNetwork(neuralNetwork* network, matrix* input, matrix* output, matrix* intermediates);

neuralNetwork* optimizeNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings);

neuralNetwork* backPropNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings);
neuralNetwork* swarmOptimizeNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings);
neuralNetwork* annealNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings);
neuralNetwork* evolveNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings);
neuralNetwork* gradientClimbNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings);

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
