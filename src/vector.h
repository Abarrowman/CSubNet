#ifndef VECTOR_H_
#define VECTOR_H_

#include <stdio.h>
#include "coreDefs.h"

struct vectorStub;
typedef struct vectorStub {
	netF* vals;
	size_t length;
} vector;

vector* createVector(int length);
vector* createZeroVector(int length);

vector* normalizeVector(vector* vec);
vector* scaleVector(vector* vec, netF scale);


void deleteVector(vector* vec);

#endif
#pragma once
