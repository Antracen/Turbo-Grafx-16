#define BDIMX 32
#define BDIMY 16


dim3 block(BDIMX,BDIMY);
dim3 grid(1,1);

// Write global thread indices to a 2D shared memory array
// Read the values from shared memory and store them to global memory

__global__ void setRowReadRow(int *out) {
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int block_id = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned int thread_id = (block_id * blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x + threadIdx.x);

    // store in shared memory
    tile[threadIdx.y][threadIdx.x] = thread_id;

    // Wait for all threads
    __syncthreads();
    
    
    out[thread_id] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out) {
    __shared__ int tile[BDIMX][BDIMY];

    unsigned int block_id = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned int thread_id = (block_id * blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x + threadIdx.x);

        // store in shared memory
        tile[threadIdx.x][threadIdx.y] = thread_id;

        // Wait for all threads
        __syncthreads();
        
        
        out[thread_id] = tile[threadIdx.x][threadIdx.y];    
}

__global__ void setRowReadCol(int *out) {
    extern __shared__ int tile[];

    unsigned int block_id = (gridDim.x * blockIdx.y) + blockIdx.x;
    unsigned int thread_id = (block_id * blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x + threadIdx.x);

    // row and col is the transpose of (threadIdx.x,threadIdx.y)
    unsigned int col = thread_id / blockDim.y;
    unsigned int row = thread_id % blockDim.y;

    tile[thread_id] = thread_id;

    __syncthreads();

    unsigned int rowcol = row * blockDim.x + col;
    out[thread_id] = tile[rowcol];
}

int main() {
    int *d_C;
    cudaMalloc(&d_C, BDIMX*BDIMY*sizeof(int));
    cudaMemset(d_C, 0, BDIMX*BDIMY*sizeof(int));
    setRowReadCol<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
}