#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#define HEADER_SIZE 122 // BMP Header size

typedef unsigned char BYTE;

/**
 * Structure that represents a BMP image.
 */
typedef struct {
    int width;
    int height;
    float *data;
} BMPImage;

typedef struct timeval tval;

/**
 * Reads a BMP 24bpp file and returns a BMPImage structure.
 * Thanks to https://stackoverflow.com/a/9296467
 */
BMPImage readBMP(char *filename);

/**
 * Writes a BMP file in grayscale given its image data and a filename.
 */
void writeBMPGrayscale(int width, int height, float *image, char *filename);

/**
 * Checks if there has been any CUDA error. The method will automatically print
 * some information and exit the program when an error is found.
 */
void checkCUDAError();

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
inline double get_elapsed(tval t0, tval t1) {
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

/**
 * Stores the result image and prints a message.
 */
void store_result(int index, double elapsed_time, bool gpu, int width, int height, float *image);
#endif