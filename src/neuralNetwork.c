#include "neuralLayer.h"
#include "neuralNetwork.h"
#include "trainingData.h"
#include "matrix.h"
#include "utils.h"
#include "coreDefs.h"
#include "vector.h"
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
		applyLayer(network->layers[idx], old, inter , 1, NULL);
		old = inter;
	}
	applyLayer(network->layers[idx], old, output, 1, NULL);
	if (needsIntermediates) {
		deleteIntermediates(intermediates, network);
	}
	return output;
}

static netF determineAcceptanceThreshold(netF temp, netF oldErr, netF newErr) {
	return expf((oldErr - newErr) * 50 / temp);
	//return exp((oldErr - newErr) * 500 / temp);
}

static netF calculateTrainingDataError(neuralNetwork* network, trainingData* trains, matrix* output, matrix* intermediates) {
	applyNetwork(network, trains->input, output, intermediates);
	subtractMatrices(output, trains->output, output);
	return sqrtf(sumSquareMatrix(output) / (output->height * output->width));
}

static neuralNetwork* mutateNetwork(neuralNetwork* mutant, netF mutationSize) {
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
				*matVal += mutationSize * (randomNetF() - 0.5f) / 2;
			}
		}

		int biases = getLayerOutputs(layer);
		int biasIdx;
		for (biasIdx = 0; biasIdx < biases; biasIdx++) {
			layer->biases[biasIdx] += mutationSize * (randomNetF() - 0.5f);
		}
	}
	return mutant;
}

#define START_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates, initialError, bestStateError) \
neuralNetwork* current = cloneNeuralNetwork(original); \
neuralNetwork* best = cloneNeuralNetwork(original); \
neuralNetwork* mutant = cloneNeuralNetwork(original); \
matrix* output = createMatrix(train->output->height, getNetworkOutputs(original)); \
matrix* intermediates = createNetworkIntermediates(original, getTrainingDataCount(train)); \
netF initialError = calculateTrainingDataError(original, train, output, intermediates); \
PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Initial %f%%\n", initialError * 100) \
netF bestStateError = initialError;

#define FINISH_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates) \
deleteNetwork(current); \
deleteNetwork(mutant); \
deleteMatrix(output); \
deleteIntermediates(intermediates, original); \
return best;

static neuralNetwork* addVectorToNetwork(vector* change, neuralNetwork* net) {
	int layerIdx;
	int i = 0;
	for (layerIdx = 0; layerIdx < net->numLayers; layerIdx++) {
		neuralLayer* layer = net->layers[layerIdx];
		matrix* mat = layer->matrix;
		int rowIdx;
		for (rowIdx = 0; rowIdx < mat->height; rowIdx++) {
			int colIdx;
			for (colIdx = 0; colIdx < mat->width; colIdx++) {
				netF* matVal = getMatrixVal(mat, rowIdx, colIdx);
				*matVal += change->vals[i++];
			}
		}

		int biases = getLayerOutputs(layer);
		int biasIdx;
		for (biasIdx = 0; biasIdx < biases; biasIdx++) {
			layer->biases[biasIdx] += change->vals[i++];
		}
	}
	return net;
}

static int countNetworkParameters(neuralNetwork* net) {
	int n = 0;
	int paramCount = 0;
	neuralLayer* layer = NULL;
	for (n = 0; n < net->numLayers; n++) {
		layer = net->layers[n];
		paramCount += (getLayerInputs(layer) + 1) * getLayerOutputs(layer);
	}
	return paramCount;
}

static vector* copyNetworkToVector(neuralNetwork* net, vector* vec) {
	int layerIdx;
	int i = 0;
	for (layerIdx = 0; layerIdx < net->numLayers; layerIdx++) {
		neuralLayer* layer = net->layers[layerIdx];
		matrix* mat = layer->matrix;
		int rowIdx;
		for (rowIdx = 0; rowIdx < mat->height; rowIdx++) {
			int colIdx;
			for (colIdx = 0; colIdx < mat->width; colIdx++) {
				netF* matVal = getMatrixVal(mat, rowIdx, colIdx);
				vec->vals[i++] = *matVal;
			}
		}

		int biases = getLayerOutputs(layer);
		int biasIdx;
		for (biasIdx = 0; biasIdx < biases; biasIdx++) {
			vec->vals[i++] = layer->biases[biasIdx];
		}
	}
	return vec;
}

//static vector* networkAsVector(neuralNetwork* net) {
//	vector* vec = createVector(countNetworkParameters(net));
//	return copyNetworkToVector(net, vec);
//}

netOptSettings* initNetOptSettings(netOptSettings* settings) {
	settings->algorithm = 0;
	settings->maxRounds = 500000;
	settings->coolRounds = 150000;
	settings->particleCount = 100;
	settings->childCount = 1000;
	return settings;
}

neuralNetwork* optimizeNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings) {
	int alg = settings->algorithm;
	if (alg == 0) {
		return annealNetwork(original, train, settings);
	} else if (alg == 1) {
		return evolveNetwork(original, train, settings);
	} else if (alg == 2) {
		return gradientClimbNetwork(original, train, settings);
	} else if (alg == 3) {
		return swarmOptimizeNetwork(original, train, settings);
	} else if (alg == 4) {
		return backPropNetwork(original, train, settings);
	} else {
		PRINT_FLUSH(1, "Unknown algorithm [%d]\n", settings->algorithm);
		return NULL;
	}
}

matrix* applyNetworkForBackprop(neuralNetwork* network, matrix* input, matrix* output, matrix* intermediates, matrix* preOuts) {
	int idx;
	matrix* old = input;
	matrix* inter;
	for (idx = 0; idx < network->numLayers - 1; idx++) {
		inter = intermediates + idx;
		applyLayer(network->layers[idx], old, inter, 1, preOuts + idx);
		old = inter;
	}
	applyLayer(network->layers[idx], old, output, 1, preOuts + idx);
	return output;
}

neuralNetwork* backPropNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings) {
	neuralNetwork* current = cloneNeuralNetwork(original);
	matrix* output = createMatrix(train->output->height, getNetworkOutputs(original));
	matrix* extra = createMatrix(train->output->height, getNetworkOutputs(original));
	matrix* intermediates = createNetworkIntermediates(original, getTrainingDataCount(train));

	
	int numLayers = original->numLayers;
	int numTrains = getTrainingDataCount(train);
	int paramCount = countNetworkParameters(original);
	vector* changeVector = createVector(paramCount);

	int n;
	neuralLayer* layer;

	matrix* layerErrors = malloc(sizeof(matrix) * numLayers);
	matrix* weightDers = malloc(sizeof(matrix) * numLayers);
	matrix* layerExtras = malloc(sizeof(matrix) * numLayers);
	matrix* transInters = malloc(sizeof(matrix) * numLayers);
	for (n = 0; n < numLayers; n++) {
		initMatrix(layerExtras + n, numTrains, getLayerOutputs(original->layers[n]));
		initMatrix(layerErrors + n, numTrains, getLayerOutputs(original->layers[n]));
		initMatrix(weightDers + n, 1, getLayerOutputs(original->layers[n]) * getLayerInputs(original->layers[n]));

		if (n == 0) {
			initMatrix(transInters, train->input->width, train->input->height);
		} else {
			matrix* currentInter = intermediates + n - 1;
			initMatrix(transInters + n, currentInter->width, currentInter->height);
		}
	}

	netF initialError = 0;
	netF rmsError = 0;
	int round;
	for (round = 0; round < settings->maxRounds; round++) {
		applyNetworkForBackprop(current, train->input, output, intermediates, layerErrors);
		subtractMatrices(output, train->output, extra);
		netF error = sumSquareMatrix(extra) / (2 * extra->height * extra->width);
		rmsError = sqrtf(error * 2);
		if (round == 0) {
			initialError = rmsError;
			PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Initial %f%%\n", initialError * 100)
		} else if (round % 1000 == 0){
			PRINT_FLUSH(1, "Cycle %d Initial %f%% Current %f%% \n", round, initialError * 100, rmsError * 100)
		}

		matrix* layExtra = layerExtras + (numLayers - 1);
		subtractMatrices(output, train->output, layExtra);
		matrix* layerEr = layerErrors + (numLayers - 1);
		layer = current->layers[current->numLayers - 1];
		elementMultMatrices(layExtra, calculateLayerOutputDerrivative(layer, layerEr), layerEr);

		for (n = numLayers - 2; n >= 0; n--) {
			layExtra = layerExtras + n;
			//layer and layerEr are referring to the next layer
			multiplyMatrices(layerEr, layer->matrix, layExtra);
			layerEr = layerErrors + n;
			layer = current->layers[n];
			elementMultMatrices(layExtra, calculateLayerOutputDerrivative(layer, layerEr), layerEr);
		}

		int changeIdx = 0;
		for (n = 0; n < numLayers; n++) {
			layerEr = layerErrors + n;
			matrix* weightDer = weightDers + n;

			matrix* transLayerEr = layerExtras + n;
			matrix* transInter = transInters + n;
			transLayerEr->width = layerEr->height;
			transLayerEr->height = layerEr->width;
			transposeMatrix(layerEr, transLayerEr);

			if (n == 0) {
				transposeMatrix(train->input, transInter);
			} else {
				transposeMatrix(intermediates + n - 1, transInter);
			}

			transExpandMultCollapseMatrices(transLayerEr, transInter, weightDer);
			transLayerEr->height = layerEr->height;
			transLayerEr->width = layerEr->width;

			//these averages are slow and could be made faster
			//weights
			int col;
			for (col = 0; col < weightDer->width; col++) {
				changeVector->vals[changeIdx] = *getMatrixVal(weightDer, 0, col);
				changeIdx++;
			}
			//biases
			for (col = 0; col < layerEr->width; col++) {
				int row;
				netF sum = 0;
				for (row = 0; row < layerEr->height; row++) {
					sum += *getMatrixVal(layerEr, row, col);
				}
				changeVector->vals[changeIdx] = -sum / layerEr->height;
				changeIdx++;
			}
		}
		multiplyVectorSelf(changeVector, -0.5f);
		addVectorToNetwork(changeVector, current);
	}
	PRINT_FLUSH(1, "Cycle %d Initial %f%% Best %f%% \n", round, initialError * 100, rmsError * 100)

	deleteVector(changeVector);
	for (n = 0; n < numLayers; n++) {
		clearMatrix(layerErrors + n);
		clearMatrix(layerExtras + n);
		clearMatrix(weightDers + n);
	}
	free(layerExtras);
	free(weightDers);
	free(layerErrors);
	deleteIntermediates(intermediates, original);
	deleteMatrix(output);
	deleteMatrix(extra);
	return current;
}

neuralNetwork* swarmOptimizeNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings) {
	neuralNetwork* best = cloneNeuralNetwork(original);
	matrix* output = createMatrix(train->output->height, getNetworkOutputs(original));
	matrix* intermediates = createNetworkIntermediates(original, getTrainingDataCount(train));
	netF initialError = calculateTrainingDataError(original, train, output, intermediates);
	PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Initial %f%%\n", initialError * 100)
	netF bestStateError = initialError;

	netF searchArea = 20;
	netF friction = 0.7f;
	netF startAttraction = 1;
	netF bestAttraction = 2;
	int maxCycles = settings->maxRounds;


	int paramCount = countNetworkParameters(original);
	vector* bestLoc = copyNetworkToVector(original, createVector(paramCount));

	int numParticles = settings->particleCount;


	//consider using a struct instead
	neuralNetwork** particles = malloc(sizeof(neuralNetwork*) * numParticles);
	vector** starts = malloc(sizeof(vector*) * numParticles);
	vector** locations = malloc(sizeof(vector*) * numParticles);
	vector** velocities = malloc(sizeof(vector*) * numParticles);

	int n;
	neuralNetwork* mutant;
	for (n = 0; n < numParticles; n++) {
		mutant = mutateNetwork(cloneNeuralNetwork(original), searchArea);
		particles[n] = mutant;
		vector* startPos = copyNetworkToVector(mutant, createVector(paramCount));
		starts[n] = startPos;
		locations[n] = cloneVector(startPos);
		velocities[n] = createZeroVector(paramCount);
		netF mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);
		if (mutantStateError < bestStateError) {
			bestStateError = mutantStateError;
			copyNeuralNetwork(mutant, best);
			copyVector(startPos, bestLoc);
		}
	}

	int cycle;
	vector* temp = createVector(paramCount);
	for (cycle = 0; cycle < maxCycles; cycle++) {
		int particleIdx = cycle % numParticles;
		vector* velocity = velocities[particleIdx];
		multiplyVectorSelf(velocity, friction);
		vector* location = locations[particleIdx];
		subVectors(starts[particleIdx], location, temp);
		multiplyVectorSelf(temp, startAttraction * randomNetF());
		addVectorToSelf(velocity, temp);
		subVectors(bestLoc, location, temp);
		multiplyVectorSelf(temp, bestAttraction * randomNetF());
		addVectorToSelf(velocity, temp);

		mutant = particles[particleIdx];
		addVectorToNetwork(velocity, mutant);
		copyNetworkToVector(mutant, location);

		netF mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);
		if (mutantStateError < bestStateError) {
			bestStateError = mutantStateError;
			copyNeuralNetwork(mutant, best);
			copyVector(location, bestLoc);
		}
		if (cycle % 10000 == 0) {
			PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Cycle %d Best %f%% Mut %f%%\n", cycle,
				bestStateError * 100, mutantStateError * 100)
		}
	}
	deleteVector(temp);

	PRINT_FLUSH(1, "Cycle %d Initial %f%% Best %f%% \n", cycle, initialError * 100, bestStateError * 100)

	for (n = 0; n < numParticles; n++) {
		deleteNetwork(particles[n]);
		deleteVector(starts[n]);
		deleteVector(velocities[n]);
		deleteVector(locations[n]);
	}
	free(bestLoc);
	free(locations);
	free(velocities);
	free(starts);
	free(particles);
	deleteMatrix(output);
	deleteIntermediates(intermediates, original);
	return best;
}

neuralNetwork* gradientClimbNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings) {
	START_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates, initialError, bestStateError)

	int maxCycles = settings->maxRounds;

	int netFCount = countNetworkParameters(original);
	vector* changeVector = createVector(netFCount);
	netF currentStateError = initialError;


	int vectorIdx = 0;
	int currentLayerIdx = 0;
	int currentIdx = 0;
	int isBias = 0;
	neuralLayer* currentLayer = mutant->layers[currentLayerIdx];
	int lastIdx = getLayerInputs(currentLayer) * getLayerOutputs(currentLayer);

	int cycle;
	for (cycle = 0; cycle < maxCycles; cycle+=2) {

		//netF stepSize = initStepSize * exp(-2.0 * cycle / maxCycles) + 0.001; //3 -> 41081
		//netF stepSize = initStepSize * exp(-5.0 * cycle / maxCycles) + 0.001; //3 -> 40594
		//netF stepSize = initStepSize * exp(-7.0 * cycle / maxCycles) + 0.001; //3 -> 40379
		//netF stepSize = initStepSize * exp(-9.5 * cycle / maxCycles) + 0.001; //  3 -> ? 40427
		//netF stepSize = initStepSize * exp(-13.0 * cycle / maxCycles) + 0.001; //  3 -> 40507
		netF stepSize = 0.1f; //3 -> ???


		copyNeuralNetwork(current, mutant);

		if (isBias) {
			currentLayer->biases[currentIdx] += stepSize;
		} else {
			currentLayer->matrix->vals[currentIdx] += stepSize;
		}
		netF mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);
		if (mutantStateError < bestStateError) {
			bestStateError = mutantStateError;
			copyNeuralNetwork(mutant, best);
		}
		changeVector->vals[vectorIdx] = (currentStateError - mutantStateError) / stepSize;

		if (isBias) {
			currentLayer->biases[currentIdx] -= 2 * stepSize;
		} else {
			currentLayer->matrix->vals[currentIdx] -= 2 * stepSize;
		}
		mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);
		if (mutantStateError < bestStateError) {
			bestStateError = mutantStateError;
			copyNeuralNetwork(mutant, best);
		}
		changeVector->vals[vectorIdx] -= (currentStateError - mutantStateError) / stepSize;

		currentIdx++;
		vectorIdx++;
		if (isBias) {
			if (currentIdx == lastIdx) {
				if ((++currentLayerIdx) >= mutant->numLayers) {
					netF randVal = 1;// randomNetF();
					scaleVectorSelf(changeVector, randVal * stepSize);
					addVectorToNetwork(changeVector, current);
					//mutateNetwork(current, 0.01);

					currentStateError = calculateTrainingDataError(current, train, output, intermediates);
					if (currentStateError < bestStateError) {
						bestStateError = currentStateError;
						copyNeuralNetwork(current, best);
					}

					currentLayerIdx = 0;
					vectorIdx = 0;
				}
				//move on to the next layer
				currentIdx = 0;
				isBias = 0;
				currentLayer = mutant->layers[currentLayerIdx];
				lastIdx = getLayerInputs(currentLayer) * getLayerOutputs(currentLayer);
			}
		} else {
			if (currentIdx == lastIdx) {
				//move on to the biases
				currentIdx = 0;
				lastIdx = getLayerOutputs(currentLayer);
				isBias = 1;
			}
		}
		if (cycle % 10000 == 0) {
			PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Cycle %d Step Size %f Best %f%% Curr %f%% Mut %f%%\n", cycle, stepSize,
						bestStateError * 100, currentStateError * 100, mutantStateError * 100)
		}
	}
	PRINT_FLUSH(1, "Cycle %d Initial %f%% Best %f%% \n", cycle, initialError * 100, bestStateError * 100)
	deleteVector(changeVector);
	FINISH_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates)
}

neuralNetwork* evolveNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings) {
	START_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates, initialError, bestStateError)

	int maxCycles = settings->maxRounds;
	int childrenPerGeneration = settings->childCount;

	int cycle;
	for (cycle = 0; cycle < maxCycles; cycle++) {
		copyNeuralNetwork(current, mutant);
		mutateNetwork(mutant, 1.0f);

		netF mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);

		if (mutantStateError < bestStateError) {
			bestStateError = mutantStateError;
			copyNeuralNetwork(mutant, best);
		}
		if (cycle % childrenPerGeneration == 0) {
			copyNeuralNetwork(best, current);
		}
		if (cycle % 10000 == 0) {
			PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Cycle %d Best %f%% Mut %f%%\n", cycle, bestStateError * 100, mutantStateError * 100)
		}
	}	
	
	PRINT_FLUSH(1, "Cycle %d Initial %f%% Best %f%% \n", cycle, initialError * 100, bestStateError * 100)

	FINISH_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates)
}

neuralNetwork* annealNetwork(neuralNetwork* original, trainingData* train, netOptSettings* settings) {
	START_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates, initialError, bestStateError)

	int maxCycles = settings->maxRounds;
	int cooldownCycles = settings->coolRounds;
	netF initialTemp = 5;
	netF temp = initialTemp;
	int cyclesWorse = 0;
	
	netF currentStateError = initialError;


	int cycle;
	for (cycle = 0; cycle < maxCycles; cycle++) {
		temp = initialTemp * expf(-3.5f * cycle / cooldownCycles);
		copyNeuralNetwork(current, mutant);
		mutateNetwork(mutant, 1.0f);//(1.0f + temp) / 2.0f);

		netF mutantStateError = calculateTrainingDataError(mutant, train, output, intermediates);

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

		if (cycle % 10000 == 0) {
			PRINT_FLUSH(NEURAL_NETWORK_INCLUDE_DEBUG_LOGS, "Cycle %d Temp %f Best %f%% Curr %f%% Mut %f%%\n", cycle, temp,
					bestStateError * 100, currentStateError * 100, mutantStateError * 100)
		}
	}

	PRINT_FLUSH(1, "Cycle %d Initial %f%% Best %f%% \n", cycle, initialError * 100, bestStateError * 100)

	FINISH_OPTIMIZING_NETWORK(original, current, best, mutant, output, intermetidates)
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
	size_t encodedSize = sizeof(int);
	int layer = 0;
	for (layer = 0; layer < network->numLayers; layer++) {
		matrix* mat = network->layers[layer]->matrix;
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
