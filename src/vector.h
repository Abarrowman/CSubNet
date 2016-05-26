#ifndef VECTOR_H_
#define VECTOR_H_

#include <stdio.h>
#include "coreDefs.h"

struct vectorStub;
typedef struct vectorStub {
	netF* vals;
	size_t length;
} vector;

vector* createVector(size_t length);
vector* createZeroVector(size_t length);
vector* cloneVector(vector* vec);

vector* copyVector(vector* vec, vector* dest);

vector* normalizeVector(vector* vec, vector* output);
vector* normalizeVectorSelf(vector* vec);

vector* scaleVectorSelf(vector* vec, netF mag);
vector* scaleVector(vector* vec, netF mag, vector* output);

vector* multiplyVectorSelf(vector* vec, netF scale);
vector* multiplyVector(vector* vec, netF scale, vector* output);

vector* addVectorToSelf(vector* left, vector* right);
vector* addVectors(vector* left, vector* right, vector* dest);

vector* subVectorFromSelf(vector* left, vector* right);
vector* subVectors(vector* left, vector* right, vector* dest);

vector* fillVectorZero(vector* vec);

void deleteVector(vector* vec);

#endif
#pragma once
