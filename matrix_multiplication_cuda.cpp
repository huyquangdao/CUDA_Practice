#include<torch/extension.h>
#include<vector>

using namespace std;


torch::Tensor matrix_multiplication_forward_cuda(torch::Tensor a, torch::Tensor b);

std::vector<torch::Tensor> matrix_multiplication_backward_cuda(torch::Tensor grad_result, torch::Tensor a, torch::Tensor b);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, "must be contigous")
// #define CHECK_SHAPE(x,y) TORCH_CHECK(x.size(1) == y.size(0), "input matrices must to have proper shapes")
// #define CHECK_INPUT(x,y) CHECK_CUDA(x); CHECK_CUDA(y); CHECK_SHAPE(x,y); CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(y);

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

torch::Tensor matrix_multiplication_forward(torch::Tensor a, torch::Tensor b){
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return matrix_multiplication_forward_cuda(a,b);
}

std::vector<torch::Tensor> matrix_multiplication_backward(torch::Tensor grad_result, torch::Tensor a, torch::Tensor b){
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return matrix_multiplication_backward_cuda(grad_result, a, b);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matrix_multiplication_forward, "Multiplication forward (CUDA)");
  m.def("backward", &matrix_multiplication_backward, "Multiplication backward (CUDA)");
}