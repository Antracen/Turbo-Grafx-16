#include <sys/time.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>

#define ARRAY_SIZE (2<<28)
#define TPB 256

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

/*
	Single-precision A*X + Y for constant a and arrays X and Y
	Compute a*x + y and store result in y
*/
__global__ void saxpy_gpu(const float *x, float *y, const float a) {
	long i = (long) blockIdx.x * (long) blockDim.x + (long) threadIdx.x;
	if(i < ARRAY_SIZE) y[i] += a*x[i];
}

/*
    Single-precision A*X + Y for constant a and arrays X and Y
    Compute a*x + y and store result in y
*/
__host__ void saxpy_cpu(const float *x, float *y, const float a) {
	#pragma omp target
	#pragma omp teams distribute parallel for 
   for(int i = 0; i < ARRAY_SIZE; i++) {
		y[i] += a*x[i];
	}
}

int main() {

	double timing; // Used for timing

	/* RANDOM GENERATION */
		std::random_device rd;
		std::mt19937 engine(rd());
		std::uniform_real_distribution<float> dist(-1000,1000);

	/* TEST VARIABLES */
		float *x = new float[ARRAY_SIZE];
		float *cpu_y = new float[ARRAY_SIZE];
		float *gpu_y = new float[ARRAY_SIZE];
		float a = dist(engine);

		for(int i = 0; i < ARRAY_SIZE; i++) {
			x[i] = dist(engine);
			cpu_y[i] = dist(engine);
			gpu_y[i] = cpu_y[i];
		}		

	/* CPU TEST */
		timing = cpuSecond();
		saxpy_cpu(x, cpu_y, a);
		timing = cpuSecond() - timing;

		std::cout << "CPU SAXPY TOOK " << timing << " SECONDS." << std::endl;

	/* GPU TEST */
		timing = cpuSecond();

		float *device_x, *device_y;

		cudaMalloc(&device_x, ARRAY_SIZE*sizeof(float));
		cudaMalloc(&device_y, ARRAY_SIZE*sizeof(float));
		
		cudaMemcpy(device_x, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(device_y, gpu_y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

		/*  Num blocks are rounded upwards to make sure the integer division for
        	calculating number of blocks may result in too few blocks
    	*/
		int num_blocks = (ARRAY_SIZE+TPB-1)/TPB;
		printf("%d blocks\n", num_blocks);
		saxpy_gpu<<<num_blocks, TPB>>>(device_x, device_y, a);

		cudaMemcpy(gpu_y, device_y, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		timing = cpuSecond() - timing;

        std::cout << "GPU SAXPY TOOK " << timing << " SECONDS." << std::endl;

	/* COMPARE */
		bool good = true;
		float tol = 0.1;
		for(int i = 0; i < ARRAY_SIZE; i++) {
			if(std::abs(cpu_y[i] - gpu_y[i]) > tol) {
				good = false;
				std::cout << "ERROR, arrays differ at " << i << " by " <<  std::abs(cpu_y[i] - gpu_y[i]) << std::endl;
				std::cout << std::setprecision(10) << cpu_y[i] << " vs " << gpu_y[i] << std::endl;
				break;
			}
		}
		if(good) std::cout << "Comparison succeeded" << std::endl;

	/* CLEAN UP */
		delete[] x;
		delete[] cpu_y;
		delete[] gpu_y;
		cudaFree(device_x);
		cudaFree(device_y);

	return 0;
}
