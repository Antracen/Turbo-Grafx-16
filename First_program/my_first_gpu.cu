/*
    This program is the first program in the DD2360 course Applied GPU Programming

    The problem to solve is to calculate an array of distances from a reference point
    to each of N points uniformly spaced along a line segment

    First assign values between 0 and 1 to each point uniformly (scale).
    Then calculate the distance from ref to each point (distance)

    This uses GPU programming to do the calculations. Each point is handled by a separate thread

    Functions are prepended with qualifiers according to the following rules.
        __global__ are launched from host (CPU) and ran on device (GPU)
        __device__ are launched from device and ran on device
 
	Compile code on KTH Tegner by running:
		nvcc -arch=sm_30 my_first_gpu.cu -o my_first_gpu.out

	Run the code on KTH Tegner by running:
		sbatch runner.sh
*/


// #include <cmath> CUDA internal files already include math.h

#include <stdio.h> // Printing

#define N 64
#define TPB 32 // Threads per block


/*
    The scaling function gives the "i"th point on an interval uniformly spaced with "n" points
*/
__device__ float scale(int i, int n) {
    return ((float) i) / (n-1);
}

/*
    Calculate Euclidean distance between 2 1D-points
    Since it is launched from GPU to be run on GPU we have __device__ qualifier
*/
__device__ float distance(float x1, float x2) {
    return sqrt((x1-x2) * (x1-x2));
}




/*
    A kernel function to be ran on each GPU thread.
*/
__global__ void scale_and_distance(float *res, float ref, int len) {

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const float x = scale(i, len);
    res[i] = distance(x, ref);
    printf("i = %2d: dist from %f to %f is %f\n", i, ref, x, res[i]);

}




int main() {

    const float ref = 0.5;

    // Create an array on GPU memory to store results from each thread
    float *res;
    cudaMalloc(&res, N*sizeof(float));

    /*  Num blocks are rounded upwards to make sure the integer division for
        calculating number of blocks may result in too few blocks
    */
    int num_blocks = (N+TPB-1)/TPB;

    // Launch kernel (function) on the GPU.
    scale_and_distance<<<num_blocks, TPB>>>(res, ref, N);

    // Use synchronisation to make sure the "printf" operation finishes in all threads
    cudaDeviceSynchronize();

    // Free memory and return
    cudaFree(res);
    return 0;

}
