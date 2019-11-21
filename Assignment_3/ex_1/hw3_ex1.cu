/* 
    I moved everything that we do not need to modify
    for this assignment into a separate header file.
*/
#include "tools.hpp"

#define BLOCK_SIZE 16
#define BLOCK_SIZE_SH (BLOCK_SIZE+2)

/**
 * Converts a given 24bpp image into 8bpp grayscale using the CPU.
 */
void cpu_grayscale(int width, int height, float *image, float *image_out) {
        for (int h = 0; h < height; h++) {

            int offset_out = h*width; // 1 color per pixel
            int offset = offset_out*3; // 3 colors per pixel
            
            for (int w = 0; w < width; w++) {

                float *pixel = &image[offset + w * 3];
                
                // Convert to grayscale following the "luminance" model
                image_out[offset_out + w] = pixel[0] * 0.0722f + // B
                                            pixel[1] * 0.7152f + // G
                                            pixel[2] * 0.2126f;  // R
            }
        }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__ void gpu_grayscale(int width, int height, float *image, float *image_out) {

        int block_id = (blockIdx.y * gridDim.x) + blockIdx.x;
        int thread_id = block_id*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;

        if(thread_id >= width*height) return;
        
        float *pixel = &image[thread_id*3];

        image_out[thread_id] = pixel[0] * 0.0722f + // B
                               pixel[1] * 0.7152f + // G
                               pixel[2] * 0.2126f;  // R

}

/**
 * Applies a 3x3 convolution matrix to a pixel using the CPU.
 */
float cpu_applyFilter(float *image, int stride, float *matrix, int filter_dim) {
    
        float pixel = 0.0f;
        
        for (int h = 0; h < filter_dim; h++) {
            int offset_image = h * stride;
            int offset_matrix = h * filter_dim;
            
            for (int w = 0; w < filter_dim; w++) {
                pixel += image[offset_image + w] * matrix[offset_matrix + w];
            }
        }
        
        return pixel;
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the GPU.
 */
__device__ float gpu_applyFilter(float *sh_image, int stride, float *matrix, int filter_dim) {

        float pixel = 0.0f;

        __syncthreads();
        
        for (int h = 0; h < filter_dim; h++) {
            int offset_image = h * stride;
            int offset_matrix = h * filter_dim;
            
            for (int w = 0; w < filter_dim; w++) {
                pixel += sh_image[offset_image + w] * matrix[offset_matrix + w];
            }
        }
        
        return pixel;
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the CPU.
 */
void cpu_gaussian(int width, int height, float *image, float *image_out) {

        float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                            2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                            1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
        
        for (int h = 0; h < (height - 2); h++) {
            int offset_t = h * width;
            int offset   = (h + 1) * width;
            
            for (int w = 0; w < (width - 2); w++) {
                image_out[offset + (w + 1)] = cpu_applyFilter(&image[offset_t + w], width, gaussian, 3);
            }
        }
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU.
 */
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out) {

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_id = index_y * width + index_x;
    if(thread_id >= width*height) return;
    
    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];
    
    float gaussian[9] = {1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f, 2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f}; 
    
    if(index_x < (width - 2) && index_y < (height - 2)) {
        int offset   = (index_y + 1) * width + (index_x + 1);
        int thread_id_block = (threadIdx.y)*BLOCK_SIZE_SH + (threadIdx.x);

        sh_block[thread_id_block] = image[thread_id];

        // VERTICAL BOTTOM
        if(threadIdx.y == BLOCK_SIZE-1 || threadIdx.y == BLOCK_SIZE-2) 
            sh_block[thread_id_block + 2*BLOCK_SIZE_SH] = image[thread_id+2*width];
        // HORIZONTAL RIGHT
        if(threadIdx.x == BLOCK_SIZE-1 || threadIdx.x == BLOCK_SIZE-2) 
            sh_block[thread_id_block + 2] = image[thread_id + 2];
        // CORNER BOTTOM RIGHT
        if((threadIdx.y == BLOCK_SIZE-1 || threadIdx.y == BLOCK_SIZE-2) && (threadIdx.x == BLOCK_SIZE-1 || threadIdx.x == BLOCK_SIZE-2))
            sh_block[thread_id_block + 2*BLOCK_SIZE_SH + 2] = image[thread_id + 2*width + 2];

        image_out[offset] = gpu_applyFilter(&sh_block[thread_id_block], BLOCK_SIZE_SH, gaussian, 3);
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the CPU.
 */
void cpu_sobel(int width, int height, float *image, float *image_out) {
    
    float sobel_x[9] = { 1.0f, 0.0f, -1.0f, 
						2.0f, 0.0f, -2.0f, 
						1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f, 2.0f, 1.0f, 
						0.0f, 0.0f, 0.0f, 
						-1.0f, -2.0f, -1.0f };
    
    for (int h = 0; h < (height - 2); h++) {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++) {
            float gx = cpu_applyFilter(&image[offset_t + w], width, sobel_x, 3);
            float gy = cpu_applyFilter(&image[offset_t + w], width, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
            image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        }
    }

}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU.
 */
__global__ void gpu_sobel(int width, int height, float *image, float *image_out) {
    
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int thread_id = index_y * width + index_x;
    if(thread_id >= width*height) return;

    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];
    
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f };

    float sobel_y[9] = {1.0f, 2.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f, -2.0f, -1.0f };
        
    if (index_x < (width - 2) && index_y < (height - 2)) {
        int offset   = (index_y + 1) * width + (index_x + 1);
        int thread_id_block = (threadIdx.y)*BLOCK_SIZE_SH + (threadIdx.x);

        sh_block[thread_id_block] = image[thread_id];

        // VERTICAL BOTTOM
        if(threadIdx.y == BLOCK_SIZE-1 || threadIdx.y == BLOCK_SIZE-2) 
            sh_block[thread_id_block + 2*BLOCK_SIZE_SH] = image[thread_id+2*width];
        // HORIZONTAL RIGHT
        if(threadIdx.x == BLOCK_SIZE-1 || threadIdx.x == BLOCK_SIZE-2) 
            sh_block[thread_id_block + 2] = image[thread_id + 2];
        // CORNER BOTTOM RIGHT
        if((threadIdx.y == BLOCK_SIZE-1 || threadIdx.y == BLOCK_SIZE-2) && (threadIdx.x == BLOCK_SIZE-1 || threadIdx.x == BLOCK_SIZE-2))
            sh_block[thread_id_block + 2*BLOCK_SIZE_SH + 2] = image[thread_id + 2*width + 2];
        
        float gx = gpu_applyFilter(&sh_block[thread_id_block], BLOCK_SIZE_SH, sobel_x, 3);
        float gy = gpu_applyFilter(&sh_block[thread_id_block], BLOCK_SIZE_SH, sobel_y, 3);
        image_out[offset] = sqrtf(gx * gx + gy * gy);
    }
}

/*
    I am getting very slight differences in my GPU and CPU implementations of the greyscale function. The pixels which differ don't seem to follow any discernible pattern. I believe this is due to floating point calculations.
*/
int main(int argc, char **argv) {

    BMPImage bitmap          = { 0 };
    float    *image_out[2]   = { 0 };
    float    *d_bitmap       = { 0 };
    float    *d_image_out[2] = { 0 };
    int      image_size      = 0;
    tval     t[2]            = { 0 };
    double   elapsed_time    = 0.0f;
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change
    
    // Make sure the filename is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.width * bitmap.height;
    grid       = dim3(((bitmap.width  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((bitmap.height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));
    
    printf("Image opened (width=%d height=%d).\n", bitmap.width, bitmap.height);
    
    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++)
    {
        image_out[i] = new float[image_size];
        
        cudaMalloc(&d_image_out[i], image_size*sizeof(float));
        cudaMemset(d_image_out[i], 0, image_size*sizeof(float));
    }

    cudaMalloc(&d_bitmap, image_size * sizeof(float) * 3);
    cudaMemcpy(d_bitmap, bitmap.data, image_size * sizeof(float) * 3, cudaMemcpyHostToDevice);
    
    /* CONVERT TO GREYSCALE */

        /* CPU VERSION */
            gettimeofday(&t[0], NULL);
            cpu_grayscale(bitmap.width, bitmap.height, bitmap.data, image_out[0]);
            gettimeofday(&t[1], NULL);
            
            elapsed_time = get_elapsed(t[0], t[1]);
            store_result(1, elapsed_time, false, bitmap.width, bitmap.height, image_out[0]);
        
        /* GPU VERSION */
            gettimeofday(&t[0], NULL);
            gpu_grayscale<<<grid, block>>>(bitmap.width, bitmap.height, d_bitmap, d_image_out[0]);
            for(int i = 0; i < image_size; i++) image_out[0][i] = 0;
            cudaMemcpy(image_out[0], d_image_out[0], image_size * sizeof(float), cudaMemcpyDeviceToHost);
            gettimeofday(&t[1], NULL);

            elapsed_time = get_elapsed(t[0], t[1]);
            store_result(1, elapsed_time, true, bitmap.width, bitmap.height, image_out[0]);
    
    /* APPLY A 3x3 GAUSSIAN FILTER */

        /* CPU VERSION */
            gettimeofday(&t[0], NULL);
            cpu_gaussian(bitmap.width, bitmap.height, image_out[0], image_out[1]);
            gettimeofday(&t[1], NULL);
            
            elapsed_time = get_elapsed(t[0], t[1]);
            store_result(2, elapsed_time, false, bitmap.width, bitmap.height, image_out[1]);
        
        /* GPU VERSION */
            gettimeofday(&t[0], NULL);
            gpu_gaussian<<<grid, block>>>(bitmap.width, bitmap.height, d_image_out[0], d_image_out[1]);
            cudaMemcpy(image_out[1], d_image_out[1], image_size*sizeof(float), cudaMemcpyDeviceToHost);      
            gettimeofday(&t[1], NULL);

            elapsed_time = get_elapsed(t[0], t[1]);
            store_result(2, elapsed_time, true, bitmap.width, bitmap.height, image_out[1]);
    
    /* APPLY A SOBEL FILTER */

        /* CPU VERSION */
            gettimeofday(&t[0], NULL);
            cpu_sobel(bitmap.width, bitmap.height, image_out[1], image_out[0]);
            gettimeofday(&t[1], NULL);
            
            elapsed_time = get_elapsed(t[0], t[1]);
            store_result(3, elapsed_time, false, bitmap.width, bitmap.height, image_out[0]);
        
        /* GPU VERSION */
            gettimeofday(&t[0], NULL);
            gpu_sobel<<<grid, block>>>(bitmap.width, bitmap.height, d_image_out[1], d_image_out[0]);
            cudaMemcpy(image_out[0], d_image_out[0], image_size*sizeof(float), cudaMemcpyDeviceToHost);
            gettimeofday(&t[1], NULL);

            elapsed_time = get_elapsed(t[0], t[1]);
            store_result(3, elapsed_time, true, bitmap.width, bitmap.height, image_out[0]);
        
    /* CLEAN UP EVERYTHING */
        for (int i = 0; i < 2; i++) {
            delete[] image_out[i];
            cudaFree(d_image_out[i]);
        }
        
        free(bitmap.data);
        cudaFree(d_bitmap);

    return 0;
}

