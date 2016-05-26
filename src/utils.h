#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include "coreDefs.h"

struct stringFragmentStub;
typedef struct stringFragmentStub {
	char* start;
	size_t length;
} stringFragment;

struct intResultStub;
typedef struct intResultStub {
	int ok;
	int result;
} intResult;

#define PRINT_FLUSH(guard, format, ...) \
	if (guard) { printf(format, ## __VA_ARGS__); fflush(stdout); }

void seedRand();
netF randomNetF();
netF randomZeroCenteredNetF();
int randomInt(int min, int max);

stringFragment cStringToFragment(char* cString);

stringFragment readFileIntoMemory(char* fileName, int maxSize);
int writeFileFromMemory(char* fileName, stringFragment frag);

intResult parseInt(stringFragment* frag);
intResult parseIntFromCString(char* string);
netF parseNetF(stringFragment* frag);
netF parseNetFFromCString(char* string);

int parseCommaSeperatedInts(char* string, int* values, int maxValues);

#endif
