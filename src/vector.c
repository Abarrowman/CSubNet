#include "vector.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

vector* createVector(size_t length) {
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

vector* createZeroVector(size_t length) {
	vector* vec = createVector(length);
	if (!vec) {
		return NULL;
	}
	return fillVectorZero(vec);
}

vector* cloneVector(vector* vec) {
	vector* dest = createVector(vec->length);
	if (!dest) {
		return NULL;
	}
	return copyVector(vec, dest);
}

vector* copyVector(vector* vec, vector* dest) {
	memcpy(dest->vals, vec->vals, sizeof(netF) * dest->length);
	return dest;
}

vector* scaleVectorSelf(vector* vec, netF mag) {
	return scaleVector(vec, mag, vec);
}

vector* scaleVector(vector* vec, netF mag, vector* dest) {
	netF totalMag = 0;
	size_t i;
	for (i = 0; i < vec->length; i++) {
		totalMag += vec->vals[i] * vec->vals[i];
	}
	totalMag = sqrtf(totalMag);
	for (i = 0; i < vec->length; i++) {
		dest->vals[i] = vec->vals[i] / totalMag * mag;
	}
	return dest;
}

vector* normalizeVector(vector* vec, vector* dest) {
	return scaleVector(vec, 1, dest);
}


vector* normalizeVectorSelf(vector* vec) {
	return scaleVector(vec, 1, vec);
}

vector* multiplyVectorSelf(vector* vec, netF scale) {
	return multiplyVector(vec, scale, vec);
}

vector* multiplyVector(vector* vec, netF scale, vector* dest) {
	size_t i;
	for (i = 0; i < vec->length; i++) {
		dest->vals[i] = vec->vals[i] * scale;
	}
	return dest;
}

vector* addVectorToSelf(vector* left, vector* right) {
	return addVectors(left, right, left);
}

vector* addVectors(vector* left, vector* right, vector* dest) {
	if ((!left) ||
		(!right) ||
		(!dest)) {
		return NULL;
	}
	size_t len = left->length;
	if ((len != right->length) ||
		(len != dest->length)) {
		return NULL;
	}
	size_t i;
	for (i = 0; i < len; i++) {
		dest->vals[i] = left->vals[i] + right->vals[i];
	}
	return dest;
}

vector* subVectorFromSelf(vector* left, vector* right) {
	return subVectors(left, right, left);
}

vector* subVectors(vector* left, vector* right, vector* dest) {
	if ((!left) ||
		(!right) ||
		(!dest)) {
		return NULL;
	}
	size_t len = left->length;
	if ((len != right->length) ||
		(len != dest->length)) {
		return NULL;
	}
	size_t i;
	for (i = 0; i < len; i++) {
		dest->vals[i] = left->vals[i] - right->vals[i];
	}
	return dest;
}

vector* fillVectorZero(vector* vec) {
	memset(vec->vals, 0, sizeof(netF) * vec->length);
	return vec;
}

void deleteVector(vector* vec) {
	free(vec->vals);
	vec->vals = NULL;
	free(vec);
}

int doesVectorHaveNANs(vector* vec) {
	if (vec == NULL) {
		return 1;
	}
	size_t i;
	for (i = 0; i < vec->length; i++) {
		if (isnan(vec->vals[i])) {
			return 1;
		}
	}
	return 0;
}