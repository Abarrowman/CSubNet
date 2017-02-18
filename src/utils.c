#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include "coreDefs.h"
#include "utils.h"

void seedRand() {
	srand((unsigned int) time(NULL));
}

netF randomNetF() {
	return ((netF) rand()) / (((netF) RAND_MAX) + 1);
}

int randomInt(int min, int max) {
	return (int) (randomNetF() * (max - min + 1) + min);
}

stringFragment readFileIntoMemory(char* fileName, int maxSize) {

	stringFragment frag = { .start = NULL, .length = 0 };

	FILE* fp;
	errno_t err = fopen_s(&fp, fileName, "rb");
	if (err != 0) {
		return frag;
	}

	int fd = _fileno(fp);
	if (fd == -1) {
		return frag;
	}

	struct stat statResult;
	if (fstat(fd, &statResult)) {
		return frag;
	}

	int fileSize = statResult.st_size;
	frag.length = fileSize;

	if (fileSize > maxSize) {
		fclose(fp);
		return frag;
	}

	char* fileText = malloc(fileSize);
	if (fileText == NULL) {
		fclose(fp);
		return frag;
	}

	size_t result = fread(fileText, 1, fileSize, fp);
	if (result != fileSize) {
		fclose(fp);
		free(fileText);
		return frag;
	}
	if (fclose(fp)) {
		free(fileText);
		return frag;
	}

	frag.start = fileText;

	return frag;
}

static void parseIntInnerLoop(char c, int* val, int* sign, intResult* r, int n) {
	if ((c >= '0') && (c <= '9')) {
		c -= '0';
		*val = 10 * (*val) + c;
	} else if ((n == 0) && (c == '-')) {
		*sign = -1;
	} else {
		r->ok = 0;
	}
}

intResult parseIntFromCString(char* string) {
	intResult r;
	int n = 0;
	int val = 0;
	int sign = 1;
	char c;
	r.ok = 1;
	while((c = string[n]) && r.ok) {
		parseIntInnerLoop(c, &val, &sign, &r, n);
		n++;
	}
	r.result = val * sign;
	return r;
}

intResult parseInt(stringFragment* frag) {
	intResult r;
	size_t n;
	int val = 0;
	int sign = 1;
	r.ok = 1;
	for (n = 0; (n < frag->length) && r.ok; n++) {
		char c = frag->start[n];
		parseIntInnerLoop(c, &val, &sign, &r, n);
	}
	r.result = val * sign;
	return r;
}

static void parseNetfInnerLoop(char c, netF* val, int* isDecimal,
		netF* placeValue, netF* sign, int* ok, size_t n) {
	if ((c >= '0') && (c <= '9')) {
		c -= '0'; 
		if (*isDecimal) {
			*val += *placeValue * c;
			*placeValue *= 0.1f;
		} else {
			*val = 10 * (*val) + c;
		}
	} else if (c == '.') {
		if (*isDecimal) {
			*ok = 0;
		} else {
			*isDecimal = 1;
		}
	} else if ((n == 0) && (c == '-')) {
		*sign = -1;
	} else {
		*ok = 0;
	}
}

netF parseNetFFromCString(char* string) {
	size_t n = 0;
	netF val = 0;
	int isDecimal = 0;
	netF placeValue = 0.1f;
	netF sign = 1;
	int ok = 1;
	char c;
	while ((c = string[n]) && ok) {
		parseNetfInnerLoop(c, &val, &isDecimal, &placeValue, &sign, &ok, n);
		n++;
	}
	if (ok) {
		return val * sign;
	} else {
		return NAN;
	}
}

netF parseNetF(stringFragment* frag) {
	size_t n;
	netF val = 0;
	int isDecimal = 0;
	netF placeValue = 0.1f;
	netF sign = 1;
	int ok = 1;
	for (n = 0; (n < frag->length) && ok; n++) {
		char c = frag->start[n];
		parseNetfInnerLoop(c, &val, &isDecimal, &placeValue, &sign, &ok, n);
	}
	if (ok) {
		return val * sign;
	} else {
		return NAN;
	}
}

stringFragment cStringToFragment(char* cString) {
  stringFragment f = { .length = strlen(cString), .start = cString };
  return f;
}

int writeFileFromMemory(char* fileName, stringFragment frag) {
	FILE* fp;
	errno_t err = fopen_s(&fp, fileName, "wb");
	if (err != 0) {
		return 0;
	}
	if (fwrite(frag.start, 1, frag.length, fp) != frag.length) {
		return 0;
	}
	if (fclose(fp)) {
		return 0;
	}
	return 1;
}

int parseCommaSeperatedInts(char* string, int* values, int maxValues) {
	int idx = 0;
	int lastComma = 0;
	int valueIdx = 0;
	while (1) {
		if (string[idx] == ',' || string[idx] == '\0') {
			stringFragment frag;
			frag.length = idx - lastComma;
			frag.start = string + lastComma;
			intResult ir = parseInt(&frag);
			if (!ir.ok) {
				frag.start[frag.length] = 0;
				return -1;
			}
			if (valueIdx == maxValues) {
				return -1;
			}
			values[valueIdx] = ir.result;
			valueIdx++;
			lastComma = idx + 1;
			if (string[idx] == '\0') {
				break;
			}
		}
		idx++;
	}
	return valueIdx;
}

double clocksToSeconds(clock_t start, clock_t end) {
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}