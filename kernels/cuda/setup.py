from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="roboticattack_cuda_kernels", version="0.1.0",
    ext_modules=[CUDAExtension("roboticattack_cuda_kernels", ["robotic_attack_ops.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]})],
    cmdclass={"build_ext": BuildExtension})
