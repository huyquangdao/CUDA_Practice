import torch
import matrix_multiplication_cuda
import math
import torch

import torch.nn as nn


class CUDAMatrixMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        result = matrix_multiplication_cuda.forward(a,b)
        ctx.save_for_backward(*[a,b])
        return result

    @staticmethod
    def backward(ctx, grad_result):
        grad_a, grad_b = matrix_multiplication_cuda.backward(grad_result, *ctx.saved_tensors)
        return grad_a, grad_b


a = nn.Parameter(torch.ones(size=(3,4)).type(torch.FloatTensor).cuda())
b = nn.Parameter(torch.ones(size=(4,5)).type(torch.FloatTensor).cuda())

print(a.size(0))
print(b.size(1))

# c1 = CUDAMatrixMul.apply(a,b)

c1 = torch.mm(a,b)
c2 = torch.trace(c1)

c2 = c2 **2

c2.backward()

print(c2)

print(c1.grad)

print(a.grad)

print(b.grad)

### gradient computed by torch.mm
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]], device='cuda:0')
# tensor([[1., 1., 1., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 0., 0.]], device='cuda:0')

### gradient computed by cuda backward matrix multiplication
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]], device='cuda:0')
# tensor([[1., 1., 1., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 0., 0.]], device='cuda:0')