from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matrix_multiplication_cuda',
    ext_modules=[
        CUDAExtension('matrix_multiplication_cuda', [
            'matrix_multiplication_cuda.cpp',
            'matrix_multiplication_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
})