#include <stdio.h>

inline cudaError_t checkCuda(cudaError_t result) {
    #if defined(DEBUG) || defined(_DEBUG)
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            assert(result == cudaSuccess);
        }
    #endif
    return result;
}
 
__global__ void kernel(float *a, int offset) {

    int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    float x = (float) i;
    float s = sinf(x); 
    float c = cosf(x);
    a[i] += sqrtf(s*s+c*c);

}

int main(int argc, char **argv) {

    const int TPB = 256;
    const int num_streams = 4;
    const int stream_size = 1024*sizeof(float)*TPB;
    const int N = stream_size * num_streams;
    const int streamBytes = stream_size * sizeof(float);

    float *a, *d_a;
    checkCuda(cudaHostAlloc(&a, N*sizeof(float), cudaHostAllocDefault));
    checkCuda(cudaMalloc(&d_a, N*sizeof(float)));


    cudaStream_t stream[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        checkCuda(cudaStreamCreate(&stream[i]));
    }

    /* SEQUENTIAL */
        memset(a, 0, N*sizeof(float));
        checkCuda(cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
        kernel<<<N/TPB, TPB>>>(d_a, 0);
        checkCuda(cudaMemcpy(a, d_a, N*sizeof(float), cudaMemcpyDeviceToHost));

    /* VERSION 1 */
        memset(a, 0, N*sizeof(float));
        for (int i = 0; i < num_streams; ++i) {
            int offset = i * stream_size;
            checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]) );
            kernel<<<stream_size/TPB, TPB, 0, stream[i]>>>(d_a, offset);
            checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]) );
        }

    /* VERSION 2 */
        memset(a, 0, N*sizeof(float));
        for (int i = 0; i < num_streams; ++i)
        {
            int offset = i * stream_size;
            checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]) );
        }
        for (int i = 0; i < num_streams; ++i)
        {
            int offset = i * stream_size;
            kernel<<<stream_size/TPB, TPB, 0, stream[i]>>>(d_a, offset);
        }
        for (int i = 0; i < num_streams; ++i)
        {
            int offset = i * stream_size;
            checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]) );
        }

    for (int i = 0; i < num_streams; ++i) {
        checkCuda( cudaStreamDestroy(stream[i]) );
    }
    cudaFree(d_a);
    cudaFreeHost(a);

    return 0;
}