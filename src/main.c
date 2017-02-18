#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include "coreDefs.h"
#include "matrix.h"
#include "neuralNetwork.h"
#include "trainingData.h"
#include "utils.h"
#include "csv.h"

#define MAX_NETWORK_SIZES (10)

void help() {
	PRINT_FLUSH(1,
		"CSubNet\n"
		"\n"
		"Usage:\n"
		"Help: CSubNet -h\n"
		"Train: CSubNet -t [OPTIONS] <training-data.csv> <input-columns> <output.net>\n"
		"Re-Train: CSubNet -r [OPTIONS] <old.net> <new-training-data.csv> <output.net>\n"
		"Execute: CSubNet -e [OPTIONS] <old.net> <new-data.csv> <output.csv>\n"
		"\n"
		"Options:\n"
		"Rounds: -r=n\n"
		"Layer Sizes: -s=v,w,x,y,z\n"
		"Label Columns: -l=x\n"
		"Portion: -p=n\n"
		"Algorithm: -a=[anneal,evolve,gradient,swarm,backprop]\n")
	PAUSE();
}

intResult parseIntParam(int paramLen, char* param) {
	intResult ir;
	if (paramLen <= 2 || param[2] != '=') {
		PRINT_FLUSH(1, "Failed to parse option [%s].", param);
		ir.ok = 0;
		return ir;
	}
	char* roundString = param + 3;
	ir = parseIntFromCString(roundString);
	if (!ir.ok) {
		PRINT_FLUSH(1, "Failed to parse string [%s] as an integer.", roundString);
	}
	return ir;
}

int parseStandardAlgorithmOptions(char optChar, int paramLen, char* param, netOptSettings* optSettings) {
	intResult ir;
	if (optChar == 'a') {
		if (paramLen <= 2 || param[2] != '=') {
			PRINT_FLUSH(1, "Failed to parse option [%s].", param);
			return -1;
		}
		char* algorithmString = param + 3;
		if (!strcmp(algorithmString, "anneal")) {
			optSettings->algorithm = 0;
		} else if (!strcmp(algorithmString, "evolve")) {
			optSettings->algorithm = 1;
		} else if (!strcmp(algorithmString, "gradient")) {
			optSettings->algorithm = 2;
		} else if (!strcmp(algorithmString, "swarm")) {
			optSettings->algorithm = 3;
		} else if (!strcmp(algorithmString, "backprop")) {
			optSettings->algorithm = 4;
		} else {
			PRINT_FLUSH(1, "Failed to parse algorithm [%s].", param);
			return -1;
		}
		return 1;
	} else if (optChar == 'r') {
		ir = parseIntParam(paramLen, param);
		if (!ir.ok) {
			return -1;
		}
		optSettings->maxRounds = ir.result;
		return 1;
	} else if (optChar == 'p') {
		ir = parseIntParam(paramLen, param);
		if (!ir.ok) {
			return -1;
		}
		optSettings->portionTrain = ir.result;
		return 1;
	} else if (optChar == 'l') {
		ir = parseIntParam(paramLen, param);
		if (!ir.ok) {
			return -1;
		}
		optSettings->labelColumns = ir.result;
		return 1;
	} else {
		return 0; //unknown parameter
	}
}

int train(int paramCount, char** params) {
	char* trainCSVFile = NULL;
	int inputColumns;
	char* outputNetFile = NULL;
	int numSizes = 0;
	int layerSizes[MAX_NETWORK_SIZES];
	int defaultsFilled = 0;
	netOptSettings optSettings;
	intResult ir;

	initNetOptSettings(&optSettings);

	int paramIdx;
	for (paramIdx = 0; paramIdx < paramCount; paramIdx++) {
		char* param = params[paramIdx];
		if (param[0] == '-') {
			size_t paramLen = strlen(param);
			char optChar = param[1];
			if (optChar == 's') {
				if (paramLen <= 2 || param[2] != '=') {
					PRINT_FLUSH(1, "Failed to parse option [%s].", param)
						help();
					return 1;
				}
				char* layerSizeString = param + 3;
				numSizes = parseCommaSeperatedInts(layerSizeString, layerSizes,
					MAX_NETWORK_SIZES);
				if (numSizes == -1) {
					PRINT_FLUSH(1, "Failed to parse layer sizes [%s].",
						layerSizeString)
						help();
					return 1;
				}
			} else {
				int result = parseStandardAlgorithmOptions(optChar, paramLen, param, &optSettings);
				if (result == -1) {
					help();
					return 1;
				} else if (result == 0) {
					PRINT_FLUSH(1, "Unknown option [%s].\n", param)
						help();
					return 1;
				}
			}
		} else {
			//we've passed the options
			if (paramCount - paramIdx != 3) {
				help();
				return 1;
			}
			trainCSVFile = params[paramIdx];
			ir = parseIntFromCString(params[paramIdx + 1]);
			if (!ir.ok) {
				PRINT_FLUSH(1, "Failed to parse string [%s] as an integer.",
						params[1])
				help();
				return 1;
			}
			inputColumns = ir.result;
			outputNetFile = params[paramIdx + 2];
			defaultsFilled = 1;
			break;
		}
	}
	if (!defaultsFilled) {
		help();
		return 1;
	}

	trainingData trainData, *trainDataPtr;
	trainDataPtr = &trainData;
	if (readTrainingData(trainDataPtr, trainCSVFile, inputColumns,
			optSettings.labelColumns) == NULL) {
		PRINT_FLUSH(1, "An error occurred while reading the CSV.\n")
		help();
		return 1;
	}

	neuralNetwork* network;
	if (numSizes == 0) {
		network = createStandardNetwork(getTrainingDataIntputs(trainDataPtr),
				getTrainingDataOutputs(trainDataPtr));
	} else {
		if ((getTrainingDataIntputs(trainDataPtr) != layerSizes[0])
				|| (getTrainingDataOutputs(trainDataPtr) != layerSizes[numSizes - 1])) {
			PRINT_FLUSH(1,
					"The layer sizes provided do not match the dimensions of the CSV.\n")
			clearTrainingData(trainDataPtr);
			help();
			return 1;
		}
		network = createSizedNetwork(layerSizes, numSizes - 1);
	}

	neuralNetwork* optimalNetwork = optimizeNetwork(network, trainDataPtr, &optSettings);
	//printNeuralNetwork(optimalNetwork);
	stringFragment encoded = encodeNeuralNetwork(optimalNetwork);
	int errCode = 0;
	if (!writeFileFromMemory(outputNetFile, encoded)) {
		PRINT_FLUSH(1, "An error occurred while saving the network to [%s].\n",
				outputNetFile)
		errCode = 1;
	}

	free(encoded.start);
	deleteNetwork(optimalNetwork);
	deleteNetwork(network);
	clearTrainingData(trainDataPtr);
	return errCode;
}

int reTrain(int paramCount, char** params) {
	char* inputNetFile = NULL;
	char* trainCSVFile = NULL;
	char* outputNetFile = NULL;
	int defaultsFilled = 0;
	netOptSettings optSettings;
	int paramIdx;
	intResult ir;

	initNetOptSettings(&optSettings);

	for (paramIdx = 0; paramIdx < paramCount; paramIdx++) {
		char* param = params[paramIdx];
		if (param[0] == '-') {
			int result = parseStandardAlgorithmOptions(param[1], strlen(param), param, &optSettings);
			if (result == -1) {
				help();
				return 1;
			} else if (result == 0) {
				PRINT_FLUSH(1, "Unknown option [%s].\n", param)
				help();
				return 1;
			}
		} else {
			//we've passed the options
			if (paramCount - paramIdx != 3) {
				help();
				return 1;
			}
			inputNetFile = params[paramIdx];
			trainCSVFile = params[paramIdx + 1];
			outputNetFile = params[paramIdx + 2];
			defaultsFilled = 1;
			break;
		}
	}
	if (!defaultsFilled) {
		help();
		return 1;
	}

	neuralNetwork* network = readNeuralNetwork(inputNetFile);
	if (!network) {
		PRINT_FLUSH(1, "An error occurred while loading the network from [%s].",
				outputNetFile)
		help();
		return 1;
	}

	trainingData trainData, *trainDataPtr;
	trainDataPtr = &trainData;
	if (readTrainingData(trainDataPtr, trainCSVFile, getNetworkInputs(network),
			optSettings.labelColumns) == NULL) {
		deleteNetwork(network);
		PRINT_FLUSH(1, "An error occurred while reading the CSV.")
		help();
		return 1;
	}

	neuralNetwork* optimalNetwork = optimizeNetwork(network, trainDataPtr, &optSettings);
	stringFragment encoded = encodeNeuralNetwork(optimalNetwork);
	int errCode = 0;
	if (!writeFileFromMemory(outputNetFile, encoded)) {
		PRINT_FLUSH(1, "An error occurred while saving the network to [%s].",
				outputNetFile)
		errCode = 1;
	}

	free(encoded.start);
	deleteNetwork(optimalNetwork);
	deleteNetwork(network);
	clearTrainingData(trainDataPtr);
	return errCode;
}

int execute(int paramCount, char** params) {
	int defaultsFilled = 0;
	char* netFile = NULL;
	char* inputCSVFile = NULL;
	char* outputCSVFile = NULL;
	int paramIdx;

	netOptSettings optSettings;
	initNetOptSettings(&optSettings);

	for (paramIdx = 0; paramIdx < paramCount; paramIdx++) {
		char* param = params[paramIdx];
		if (param[0] == '-') {
			int result = parseStandardAlgorithmOptions(param[1], strlen(param), param, &optSettings);
			if (result == -1) {
				help();
				return 1;
			} else if (result == 0) {
				PRINT_FLUSH(1, "Unknown option [%s].\n", param)
					help();
				return 1;
			}
		} else {
			//we've passed the options
			if (paramCount - paramIdx != 3) {
				help();
				return 1;
			}
			netFile = params[paramIdx];
			inputCSVFile = params[paramIdx + 1];
			outputCSVFile = params[paramIdx + 2];
			defaultsFilled = 1;
			break;
		}
	}

	if (!defaultsFilled) {
		help();
		return 1;
	}

	neuralNetwork* network = readNeuralNetwork(netFile);
	if (!network) {
		PRINT_FLUSH(1, "An error occurred while loading the network from [%s].",
				netFile)
		help();
		return 1;
	}

	csv* csvData = readCSV(inputCSVFile);
	if (!csvData) {
		deleteNetwork(network);
		PRINT_FLUSH(1, "An error occurred while reading the CSV.")
		help();
		return 1;
	}
	int networkInputs = getNetworkInputs(network);
	matrix* mat = extractMatrixFromCSV(csvData, 0, optSettings.labelColumns, networkInputs,
			csvData->rows);
	if (!mat) {
		deleteCSV(csvData);
		PRINT_FLUSH(1, "An error occurred while reading the CSV.")
		help();
		return 1;
	}

	matrix* out = applyNetwork(network, mat, NULL, NULL);
	deleteNetwork(network);

	matrix* augmented = agumentMatrix(mat, out, createMatrix(mat->height, mat->width + out->width));
	deleteMatrix(mat);
	deleteMatrix(out);

	int errCode = 0;
	//writeMatrixCSV(stdout, augmented, csvData, labelColumns, networkInputs, out->width);
	if (!writeMatrixAsCSVToFile(outputCSVFile, augmented, csvData,
			optSettings.labelColumns, networkInputs, out->width)) {
		errCode = 1;
		PRINT_FLUSH(1, "An error occurred while saving the results to [%s].",
				outputCSVFile)
	}

	deleteMatrix(augmented);
	deleteCSV(csvData);
	return errCode;
}

int testCandidate(int candidate) {
	int loops = 10000;

	int leftHeight = 500;
	int rightWidth = 500;
	int leftWidth = 500;

	//int leftHeight = 2;
	//int rightWidth = 2;
	//int leftWidth = 3;

	matrix* a = createMatrix(leftHeight, leftWidth);
	matrix* at = createMatrix(leftWidth, leftHeight);
	//netF aVals[] = { 1, 2, 3,
	//				 3, 6, 9};
	//setMatrixValues(a, aVals);
	fillMatrixRandom(a);

	matrix* b = createMatrix(leftHeight, rightWidth);
	matrix* bt = createMatrix(rightWidth, leftHeight);
	//netF bVals[] = { 1, 2,
	//				 2, 4 };
	//setMatrixValues(b, bVals);
	fillMatrixRandom(b);

	matrix* d = NULL;
	if (candidate == 1 || candidate == 2) {
		d = createMatrix(1, leftWidth * rightWidth);
	} else if (candidate == 3) {
		d = createMatrix(leftHeight, leftHeight);
	} else if (candidate == 4) {
		d = createMatrix(1, 1);
	}
	
	clock_t start = clock();
	int n;
	for (n = 0; n < loops; n++) {
		if (candidate == 1) {
			expandMultCollapseMatrices(a, b, d);
		} else if (candidate == 2) {
			transposeMatrix(a, at);
			transposeMatrix(b, bt);
			transExpandMultCollapseMatrices(at, bt, d);
		} else if (candidate == 3) {
			cpuTransMultiplyMatrices(a, b, d);
		} else if (candidate == 4) {
			transposeMatrix(a, b);
		}
	}
	clock_t end = clock();
	PRINT_FLUSH(1, "CPU %lfs\n", clocksToSeconds(start, end));

	deleteMatrix(a);
	deleteMatrix(b);
	deleteMatrix(at);
	deleteMatrix(bt);
	deleteMatrix(d);
	return 0;
}

//-t - l = 1 - s = 360, 60, 20, 1 - a = backprop - r = 1 E:\A\Dropbox\Dev\Multiple\SubNet\test\malname\sanity - list.csv 360 E:\A\Dropbox\Dev\Multiple\SubNet\test\malname\network.net
//-t -r=10000 -l=2 -s=157,3,1 -a=backprop E:\A\Dropbox\Dev\Multiple\SubNet\test\mal\rated-list.csv 157 E:\A\Dropbox\Dev\Multiple\SubNet\test\mal\network-few-hidden.net
int main(int argc, char *argv[]) {
	int n;
	for (n = 0; n < argc; n++) {
		//PRINT_FLUSH(1, "Argument %d [%s]\n", n, argv[n])
		PRINT_FLUSH(1, "\"%s\" ", argv[n])
	}
	PRINT_FLUSH(1, "\n")
	int result;
	if (argc < 2) {
		help();
		result = 1;
	} else {
		char* command = argv[1];
		if (strcmp(command, "-t") == 0) {
			result = train(argc - 2, &argv[2]);
		} else if (strcmp(command, "-r") == 0) {
			result = reTrain(argc - 2, &argv[2]);
		} else if (strcmp(command, "-e") == 0) {
			result = execute(argc - 2, &argv[2]);
		} else if (strcmp(command, "-c") == 0) {
			return 	testCandidate(4);
		} else {
			help();
			if (strcmp(command, "-h") == 0) {
				result = 1;
			}
			result = 0;
		}
	}
	//PAUSE();
	return result;
}
