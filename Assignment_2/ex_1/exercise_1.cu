/*
	This program launches a kernel on 256 GPU threads
	Author: Martin Wass
*/

#include <stdio.h> // printf

// __global__ qualifier states the kernel is launched by host and ran on device
__global__ void greeting() {
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;
	printf("Hey world, from thread %d\n", threadnum);
}

int main() {
	greeting<<<1, 256>>>(); // Launch kernel on 256 threads per block and 1 block
	cudaDeviceSynchronize();
	return 0;
}
