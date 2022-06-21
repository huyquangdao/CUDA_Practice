#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cooperative_groups.h>

#define BLOCKSIZE 32

using namespace std;

template<typename scalar_t>
__global__ void matrix_multiplication_forward_cuda_kernel(
    const scalar_t* __restrict__ ma, 
    const scalar_t* __restrict__ mb,
    scalar_t* __restrict__ mc, 
    size_t M,
    size_t N, 
    size_t P){

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


template<typename scalar_t>
__global__ void matrix_multiplication_backward_cuda_kernel_a(
    scalar_t*  grad_a, 
    const scalar_t* grad_result,
    const scalar_t*  mb, 
    int M, int N, int P
){
    // global index of the current thread
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // // local indexes of the current thread.
    // int ly = threadIdx.y; 
    // int lx = threadIdx.x;

    float temp_a = 0.f;
    // for over columns of the grad result matrix.
    for (int k =0; k < P; k++){
        // da[row][col] = sum op grad_result[row][k] * mb[k][col]
        // grad_result of current col and the k-th col, mb of (col) row and  
        temp_a += grad_result[row * P + k] * mb[col * P + k];
    }

    if (row >= M || col >= N ) return;
    grad_a[row * N + col] = temp_a;

}

template<typename scalar_t>
__global__ void matrix_multiplication_backward_cuda_kernel_b(
    scalar_t*  grad_b, 
    const scalar_t* grad_result,
    const scalar_t*  ma, 
    int M, int N, int P
){
    // global index of the current thread
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // local indexes of the current thread.
    int ly = threadIdx.y; 
    int lx = threadIdx.x;

    float temp_b = 0.f;
    // for over columns of the grad result matrix.
    for (int k =0; k < M; k++){
        // da[row][col] = sum op grad_result[row][k] * mb[k][col]
        // grad_result of current col and the k-th col, mb of (col) row and  
        temp_b += grad_result[k * P + col] * ma[k * N + row];
    }

    if (row >= N || col >= P ) return;
    grad_b[row * P + col] = temp_b;

}


torch::Tensor matrix_multiplication_forward_cuda(torch::Tensor a, torch::Tensor b){

    int M = a.size(0);
    int N = a.size(1);
    int P = b.size(1);

    auto options = a.options();
    auto result = torch::zeros({M,P}, options);
    
    const dim3 BLOCKDIM(BLOCKSIZE, BLOCKSIZE);        
    const dim3 GRIDDIM( (max(a.size(0), b.size(1)) + BLOCKSIZE - 1)/ BLOCKSIZE,  (max(a.size(0), b.size(1)) + BLOCKSIZE - 1)/ BLOCKSIZE );

    /// modifying values of result by calling the cuda kernel
    AT_DISPATCH_FLOATING_TYPES(a.type(), "matrix_multiplication_cuda", ([&] {
        matrix_multiplication_forward_cuda_kernel<scalar_t><<<GRIDDIM, BLOCKDIM>>>(
            a.data<scalar_t>(),
            b.data<scalar_t>(),
            result.data<scalar_t>(),
            M, 
            N, 
            P);
    }));

    return result;
}

std::vector<torch::Tensor> matrix_multiplication_backward_cuda(torch::Tensor grad_result, torch::Tensor a, torch::Tensor b){

    int M = a.size(0);
    int N = a.size(1);
    int P = b.size(1);

    // const auto options = a.options();

    auto grad_a = torch::zeros_like(a);
    auto grad_b = torch::zeros_like(b);

    const dim3 BLOCKDIM(BLOCKSIZE, BLOCKSIZE);        
    const dim3 GRIDDIM( (max(a.size(0), b.size(1)) + BLOCKSIZE - 1)/ BLOCKSIZE,  (max(a.size(0), b.size(1)) + BLOCKSIZE - 1)/ BLOCKSIZE  );

    AT_DISPATCH_FLOATING_TYPES(grad_result.type(), "matrix_multiplication_cuda", ([&] {
        matrix_multiplication_backward_cuda_kernel_a<scalar_t><<<GRIDDIM, BLOCKDIM>>>(
            grad_a.data<scalar_t>(),
            grad_result.data<scalar_t>(),
            b.data<scalar_t>(),
            M,
            N, 
            P);
    }));

    AT_DISPATCH_FLOATING_TYPES(grad_result.type(), "matrix_multiplication_cuda", ([&] {
        matrix_multiplication_backward_cuda_kernel_b<scalar_t><<<GRIDDIM, BLOCKDIM>>>(
            grad_b.data<scalar_t>(),
            grad_result.data<scalar_t>(),
            a.data<scalar_t>(),
            M,
            N, 
            P);
    }));

    return {grad_a, grad_b};
}