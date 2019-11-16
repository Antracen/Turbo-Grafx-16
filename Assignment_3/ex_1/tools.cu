#include "tools.hpp"

BYTE g_info[HEADER_SIZE]; // Reference header

BMPImage readBMP(char *filename) {

    BMPImage bitmap = {0};
    int image_size;
    BYTE *data = NULL;

    // Open the file, reading in binary mode.
    FILE *fp = fopen(filename, "rb");
    
    // Read the header (expected BGR - 24 bit color per pixel)
    fread(g_info, sizeof(BYTE), HEADER_SIZE, fp);

    // Get the image width / height from the header
    bitmap.width = *((int*) &g_info[18]);
    bitmap.height = *((int*) &g_info[22]);
    image_size = *((int*) &g_info[34]);
    
    // Read the image data
    data = (BYTE*) malloc(image_size*sizeof(BYTE));
    fread(data, sizeof(BYTE), image_size, fp);
    
    // Convert the pixel values to float
    bitmap.data = (float*) malloc(image_size*sizeof(float));
    
    for (int i = 0; i < image_size; i++) {
        bitmap.data[i] = (float) data[i];
    }

    fclose(fp);
    free(data);

    return bitmap;
}

void writeBMPGrayscale(int width, int height, float *image, char *filename) {
    
    FILE *file = NULL;
    
    file = fopen(filename, "wb");
    
    // Write the reference header
    fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file);
    
    // Unwrap the 8-bit grayscale into a 24bpp (for simplicity)
    for (int h = 0; h < height; h++) {

        int offset = h * width;
        
        for (int w = 0; w < width; w++) {

            BYTE pixel = (BYTE)((image[offset + w] > 255.0f) ? 255.0f :
                                (image[offset + w] < 0.0f)   ? 0.0f   :
                                image[offset + w]);
            
            // Repeat the same pixel value for BGR
            fputc(pixel, file);
            fputc(pixel, file);
            fputc(pixel, file);
        }
    }
    
    fclose(file);
}

void checkCUDAError() {
    cudaError_t cudaError = cudaGetLastError();
    
    if(cudaError != cudaSuccess)
    {
        printf("CUDA Error: Returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(-1);
    }
}

void store_result(int index, double elapsed_time, bool gpu, int width, int height, float *image) {
    char path[255];

    if(gpu) sprintf(path, "images/hw3_result_gpu_%d.bmp", index);
    else sprintf(path, "images/hw3_result_cpu_%d.bmp", index);

    writeBMPGrayscale(width, height, image, path);
    
    if(gpu) printf("Elapsed GPU: %fms / ", elapsed_time);
    else printf("Elapsed CPU: %fms / ", elapsed_time);
    
    printf("Step #%d Completed - Result stored in \"%s\".\n", index, path);

}