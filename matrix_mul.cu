#include<iostream>

using namespace std;

template<typename scalar_t>
__global__ void matrix_mul(scalar_t* mc, const scalar_t* ma, const scalar_t* mb, int M, int N, int P){
    // current thread index
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    // if the current thread index is smaller then the size of the resulted matrix.
    if (index < M * P){
        
        int i = index / P;
        int j = index % P;

        float temp = 0;
        for (int k = 0; k < N; k++){
            temp = temp + ma[i * N + k] * mb[k * N + j];
        }
        mc[index] = temp;

    } 
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

    //// sizes of matrices4
    int M = 2;
    int N = 3;
    int P = 4;
    
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
    int nThreads = 1024;
    int nBlocks = (M * P + nThreads - 1)/ nThreads;

    /// call the function.
    matrix_mul<float><<<nBlocks, nThreads>>>(d_mc, d_ma, d_mb, M, N, P);

    cudaDeviceSynchronize();
    // copy values from device to host
    cudaMemcpy(mc, d_mc, size3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    //check
    print_matrix(ma, M, N);
    print_matrix(mb, N, P);
    print_matrix(mc, M, P);

    //numerical checking
    float* cpu_C = new float[M * P];

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

    print_matrix(cpu_C, M, P);

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