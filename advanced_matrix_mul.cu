#include<iostream>

using namespace std;

#define BLOCKSIZE 32

template<typename scalar_t>
__global__ void matrix_mul(scalar_t* mc, const scalar_t* ma, const scalar_t* mb, int M, int N, int P){
    
    // global index of the current thread
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // local indexes of the current thread.
    int ly = threadIdx.y; 
    int lx = threadIdx.x;

    //shared memories for each block
    __shared__ scalar_t la[BLOCKSIZE][BLOCKSIZE];
    __shared__ scalar_t lb[BLOCKSIZE][BLOCKSIZE];

    /// variable for accumulating.
    float temp = 0;
    /// loop over all blocks
    for (int i = 0; i < (BLOCKSIZE + N - 1)/ BLOCKSIZE; i++){
        // i= block_idx
        //save elements from global memory to local memory
        if ((BLOCKSIZE * i + lx < N) && (row < M))
            la[ly][lx] = ma[(row * N) + (BLOCKSIZE * i) + lx]; // convert 2-d coordinates to 1-d coordinates.
        else 
            la[ly][lx] = 0;
        
        if ((BLOCKSIZE * i + ly < N) && (col < P))
            lb[ly][lx] = mb[(BLOCKSIZE * i + ly) * P + col]; // convert 2-d coordinates to 1-d coordinates
        else
            lb[ly][lx] = 0;

        /// synchronize all threads.
        __syncthreads();

        /// loop over each block to accumalate the values.
        for (int k = 0; k < BLOCKSIZE; k++){
            temp += la[ly][k] * lb[k][lx];
        }
        __syncthreads();
    }

    //invalid threads. global indexes go beyond the boundary.
    if (row >= M || col >= P ) return;
    mc[row * P + col] = temp;

    
}

void print_matrix(float *m, int size1, int size2){
    cout<<"------------------------------"<<endl;
    for (int i =0; i< size1;i++){
        for(int j = 0; j< size2; j++){
            cout<<m[i * size1 + j]; 
        }
        cout<<std::endl;
    }
}

int main(void){

    //// sizes of matrices
    int M = 500;
    int N = 600;
    int P = 700;
    
    int size1 = M * N * sizeof(float);
    int size2 = N * P * sizeof(float);
    int size3 = M * P * sizeof(float);
    
    /// allocate memories of the matrices
    float* ma = (float*)malloc(size1);
    float* mb = (float*)malloc(size2);
    float* mc = (float*)malloc(size3);

    // pointers in the device
    float* d_ma;
    float* d_mb;
    float* d_mc;

    //allocate memories of the device pointers.
    cudaMalloc(&d_ma, size1);
    cudaMalloc(&d_mb, size2);
    cudaMalloc(&d_mc, size3);

    /// init the values for the matrices.
    for (int i =0; i< M;i ++){
        for(int j = 0; j< N; j++){
            ma[i* N + j] = 1.0f; 
        }
    }

    for (int i =0; i< N;i ++){
        for(int j = 0; j< P; j++){
            mb[i* P + j] = 1.0f; 
        }
    }

    /// copy values from host pointers to device pointers.
    cudaMemcpy(d_ma, ma, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb, mb, size2, cudaMemcpyHostToDevice);

    //Number of threads and blocks.
    // int size = 1024;
    dim3 nThreads(BLOCKSIZE, BLOCKSIZE);
    dim3 nBlocks((max(M,P) + BLOCKSIZE - 1 )/ BLOCKSIZE, (max(M,P) + BLOCKSIZE - 1)/ BLOCKSIZE);

    /// call the function.
    matrix_mul<float><<<nBlocks, nThreads>>>(d_mc, d_ma, d_mb, M, N, P);

    cudaDeviceSynchronize();
    // copy values from device to host

    cudaMemcpy(mc, d_mc, size3, cudaMemcpyDeviceToHost);    
    //check
    // print_matrix(ma, M, N);
    // print_matrix(mb, N, P);
    // print_matrix(mc, M, P);

    //numerical checking
    float* cpu_C = new float[M* P];

    float sum;
    for (int row=0; row<M; row++){
        for (int col=0; col<P; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += ma[row*M+n]*mb[n*M+col];
            }
            cpu_C[row*M+col] = sum;
        }
    }
    // print_matrix(cpu_C, M, P);

    cout<<cpu_C[0];

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < M; ROW++){
        for (int COL=0; COL < P; COL++){
            err += cpu_C[ROW * M + COL] - mc[ROW * M + COL];
        }
    }

    cout << "Error: " << err << endl;

    // free memory
    free(ma); free(mb); free(mc); free(cpu_C);
    cudaFree(d_ma); cudaFree(d_mb); cudaFree(d_mc);

    return 0;
}