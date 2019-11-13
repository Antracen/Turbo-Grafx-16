#include <iostream>
#include <stdio.h>
#include "tools.hpp"
#include <iomanip>

#define NUM_ARGS 2
#define ATOMIC true

using std::cout;
using std::endl;

/*
    Kernel function multiplying "d_a" and "d_b", saving the result in "d_res"
    Size of "d_a" and "d_b" is "n" for both
*/
template <class T>
__global__ void dot_kernel(const T *device_a, const T *device_b, T *device_res, const size_t array_size, const size_t TPB) {
    
    const size_t i = blockDim.x*blockIdx.x + threadIdx.x;
    const size_t local_i = threadIdx.x; // Index inside the block
    
    if(i >= array_size) return; // Oversubscribed threads do nothing

    extern __shared__ T block_prods[]; // Holds the partial results for threads in block
    block_prods[local_i] = device_a[i]*device_b[i];
    __syncthreads();

    // The first thread in the block adds the block_res to the device_res
    if(local_i == 0) {
        T block_res = T();
        for(int j = 0; j < blockDim.x; j++) {
            block_res += block_prods[j];
        }
        // printf("Block_%d, blockSum = %d\n", blockIdx.x, block_res);
        if(ATOMIC) {
            atomicAdd(device_res, block_res);
        } else {
            *device_res += block_res;
        }
    }
}

/*

*/
template <class T>
__host__ T dot_product(const T *a, const T *b, const size_t array_size, const size_t TPB) {

    T *device_a;
    T *device_b;
    T *device_res;

    /* Allocate memory on GPU */
        cudaMalloc(&device_a, array_size*sizeof(T));
        cudaMalloc(&device_b, array_size*sizeof(T));
        cudaMalloc(&device_res, sizeof(T));

    /* Fill the GPU memory */
        cudaMemset(device_res, T(), sizeof(T));
        cudaMemcpy(device_a, a, array_size*sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, b, array_size*sizeof(T), cudaMemcpyHostToDevice);

    dot_kernel<<<(array_size + TPB - 1)/TPB, TPB, TPB*sizeof(T)>>>(device_a, device_b, device_res, array_size, TPB);

    T res;
    cudaMemcpy(&res, device_res, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_res);

    return res;
}

template <class T>
__host__ T dot_product_CPU(const T *a, const T *b, const size_t array_size) {

    T res = T();

    for(int i = 0; i < array_size; i++) {
        res += a[i] * b[i];
    }

    return res;
}

/*

*/
__host__ int main(int argc, char **argv) {

    size_t N;
    size_t TPB;

    if(argc < NUM_ARGS + 1) {
        cout << "Too few arguments" << endl;
        return 0;
    }
    else {
        N = str2int(argv[1]);
        TPB = str2int(argv[2]);
    }

    float *a = new float[N];
    float *b = new float[N];

    float lower = -10;
    float upper = 10;
    random_array(a, N, lower, upper);
    random_array(b, N, lower, upper);

    double time = cpuSecond();
    float product_res = dot_product(a, b, N, TPB);
    time = cpuSecond() - time;
    cout << "CUDA took " << time << " seconds!" << endl;

    time = cpuSecond();
    float product_res_CPU = dot_product_CPU(a, b, N);
    time = cpuSecond() - time;
    cout << "CPU took " << time << " seconds!" << endl;

    if(float_equal(product_res, product_res_CPU, 1e-3)) cout << "Results are equal!" << endl;
    else {
        cout << "Results are not equal!" << endl;
        detailed_print_float(product_res);
        detailed_print_float(product_res_CPU);
    }

    delete[] a;
    delete[] b;
    return 0;
}
