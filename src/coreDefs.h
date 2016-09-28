#ifndef _CORE_DEFS_H
#define _CORE_DEFS_H

//typedef double netF;
typedef float netF;


#ifdef __GNUC__
typedef int errno_t;
#define fopen_s(fpp, file, mode) (*fpp=fopen(file, mode)) == NULL
#define _fileno fileno
#endif

#endif
