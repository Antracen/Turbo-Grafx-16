#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 10000000
#define TPB 256
#define ATOMIC 1 // 0 for non-atomic addition

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__
void dotKernel(int *d_res, const int *d_a, const int *d_b, int n) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= n) return;
  const int s_idx = threadIdx.x;

  __shared__ int s_prod[TPB];
  s_prod[s_idx] = d_a[idx] * d_b[idx];
  __syncthreads();

  if (s_idx == 0) {
    int blockSum = 0;
    for (int j = 0; j < blockDim.x; ++j) {
      blockSum += s_prod[j];
    }
    // printf("Block_%d, blockSum = %d\n", blockIdx.x, blockSum);
    // Try each of two versions of adding to the accumulator
    if (ATOMIC) {
      atomicAdd(d_res, blockSum);
    } else {
      *d_res += blockSum;
    }
  }
}

int dotLauncher(const int *a, const int *b, int n) {
  int *d_res;
  int *d_a = 0;
  int *d_b = 0;
  
  cudaMalloc(&d_res, sizeof(int));
  cudaMalloc(&d_a, n*sizeof(int));
  cudaMalloc(&d_b, n*sizeof(int));
  
  cudaMemset(d_res, 0, sizeof(int));
  cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);
  
  dotKernel<<<(n + TPB - 1)/TPB, TPB>>>(d_res, d_a, d_b, n);
  
  int res;
  cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(d_res);
  cudaFree(d_a);
  cudaFree(d_b);

  return res;
}

int main() {  
  int cpu_res = 0;  
  int *a = (int*)malloc(N*sizeof(int));  
  int *b = (int*)malloc(N*sizeof(int));  
  //Initialize input arrays  
  for (int i = 0; i < N; ++i) {    
    a[i] = 1;    
    b[i] = 1;  
  }  

	double t = cpuSecond();
  for (int i = 0; i < N; ++i) {    
    cpu_res += a[i] * b[i];  
  }  
	t = cpuSecond() - t;
  printf("cpu result = %f\n", t);  
	t = cpuSecond();
  int gpu_res = dotLauncher(a, b, N);  
	t = cpuSecond() - t;
  printf("gpu result = %f\n", t);  
  free(a);  
  free(b);  
  return 0;
}
