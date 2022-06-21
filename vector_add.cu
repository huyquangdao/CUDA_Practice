#include<iostream>

using namespace std;

/// The kernel
template<typename scalar_t>
__global__ void add_vec(scalar_t* c, const scalar_t* a, const scalar_t* b, int N)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

bool check(const float* a, const float* b, const float* c, const int& N){
    for(int i = 0; i<N; i++){
        if (a[i] + b[i] != c[i]){
            cout<<a[i]<<b[i]<<c[i];
            return false;
        }
    }
    return true;
}

int main(void){

    int N = 100000;

    // host pointers
    float* a; 
    a = (float*)malloc(N * sizeof(float));
    float* b; 
    b = (float*)malloc(N * sizeof(float));
    float* c; 
    c = (float*)malloc(N * sizeof(float));

    /// device pointers
    float* d_a; 
    float* d_b;
    float* d_c;

    /// allocate memory for the device pointers.
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    /// initiazlize the values, need to init the values after the initialization of device pointers ?????
    for (int i =0; i< N; i++){
        *(a + i) = 1.0f;
        *(b + i) = 2.0f;
    }

    ///copy values from host pointers to device pointers.
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    //Number of threads and blocks.
    int nThreads = 1024;
    int nBlocks = (N + nThreads - 1)/ nThreads;

    /// call the function.
    add_vec<float><<<nBlocks, nThreads>>>(d_c, d_a, d_b, N);

    // Wait for GPU to finish before accessing on host
    // cudaDeviceSynchronize();
    
    /// copy values from device pointer to host pointer.
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cout<<c[0];
    // checking
    cout<<"result:";
    cout<<check(a, b, c, N);

    ///free the memories of pointers.
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;

}