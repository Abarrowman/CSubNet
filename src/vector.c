#include "vector.h"
#include <stdlib.h>

vector* createVector(int length) {
	vector* vec = malloc(sizeof(vector));
	if (!vec) {
		return NULL;
	}
	vec->length = length;
	vec->vals = malloc(sizeof(netF) * length);
	if (!vec->vals) {
		free(vec);
		return NULL;
	}
	return vec;
}

vector* createZeroVector(int length) {
	vector* vec = createVector(length);
	if (!vec) {
		return NULL;
	}
	int i;
	for (i = 0; i < length; i++) {
		vec->vals[i] = 0;
	}

	return vec;
}


vector* normalizeVector(vector* vec) {
	netF totalMag = 0;
	int i;
	for (i = 0; i < vec->length; i++) {
		totalMag += vec->vals[i] * vec->vals[i];
	}
	totalMag = sqrt(totalMag);
	for (i = 0; i < vec->length; i++) {
		vec->vals[i] = vec->vals[i] / totalMag;
	}
	return vec;
}

vector* scaleVector(vector* vec, netF scale) {
	int i;
	for (i = 0; i < vec->length; i++) {
		vec->vals[i] *= scale;
	}
	return vec;
}

void deleteVector(vector* vec) {
	free(vec->vals);
	vec->vals = NULL;
	free(vec);
}
