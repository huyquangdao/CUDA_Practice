import argparse
import torch
from torch.autograd import gradcheck
import matrix_multiplication_cuda

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

variables = [a,b]

if gradcheck(CUDAMatrixMul.apply, variables, eps=1e-3, atol=1e-4):
    print('Ok')