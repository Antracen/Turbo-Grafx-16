// pgcc -acc -ta=tesla:cc3x,nvidia,lineinfo -Minfo=accel matrix.cpp -o code

/*
    Calculate C = A*B matrix multiplication
*/
void matmul_acc(float *C, const float *A, const float *B, int A_rows, int A_cols, int B_rows, int B_cols) {

    if(A_cols != B_rows) {
        return A_cols/0; // throw error
    }

    int C_rows = A_rows;
    int C_cols = B_cols;

    #pragma acc parallel loop copyin(A[0:A_rows*A_cols]) copyin(B[0:B_rows*B_cols]) copyout(C[0:C_rows*C_cols])
    for(int i = 0; i < C_rows; i++) {
        #pragma acc loop
        for(int j = 0; j < C_cols, j++) {
            float sum = 0.0;
            for(int k = 0; k < A_cols; k++) {
                sum += A[i*A_rows + k]*B[k*B_rows + j];
            }
            C[i*C_rows + j] = sum;
        }
    }
}

/*
    Calculate C = A*B matrix multiplication
*/
void matmul(float *C, const float *A, const float *B, int A_rows, int A_cols, int B_rows, int B_cols) {

    if(A_cols != B_rows) {
        return A_cols/0; // throw error
    }

    int C_rows = A_rows;
    int C_cols = B_cols;

    for(int i = 0; i < C_rows; i++) {
        for(int j = 0; j < C_cols, j++) {
            float sum = 0.0;
            for(int k = 0; k < A_cols; k++) {
                sum += A[i*A_rows + k]*B[k*B_rows + j];
            }
            C[i*C_rows + j] = sum;
        }
    }
}

int main() {

}