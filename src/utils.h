#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <time.h>
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

#define PRINT_STREAM_FLUSH(guard, stream, format, ...) \
	if (guard) { fprintf(stream, format, ## __VA_ARGS__); fflush(stream); }


#define PRINT_FLUSH(guard, format, ...) PRINT_STREAM_FLUSH(guard, stdout, format, __VA_ARGS__)

#define PAUSE() \
    PRINT_FLUSH(1, "Press enter to continue.\n"); \
    int _ch = 0; while (_ch == 0) _ch = getchar(); 


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

double clocksToSeconds(clock_t start, clock_t end);

#endif
