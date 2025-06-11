import os
import glob
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = 'SatSim'
if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def find_all_kernel_extension():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    ## add all extension into ext_modules
    ext_modules = []
    modules_dir = os.path.join('satsim', 'simulation')
    module_dirs = [
        directory for directory in glob.glob(os.path.join(modules_dir, "*"))
        if os.path.isdir(directory)
    ]

    for module in module_dirs:
        cpp_path = glob.glob(os.path.join(module, "*.cpp"))
        cuda_path = glob.glob(os.path.join(module, "*.cu")) if use_cuda else []
        sources = cpp_path + cuda_path
        module_name = '.'.join(module.split(os.path.sep))
        if sources:
            ext_modules.append(
                extension(
                    f"{module_name}._C",
                    sources,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    py_limited_api=py_limited_api,
                ))

    return ext_modules


setup(
    name=library_name,
    version="0.2.0dev",
    packages=find_packages(),
    ext_modules=find_all_kernel_extension(),
    install_requires=["torch"],
    description="debug for encoder kernel",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {
        "py_limited_api": "cp39"
    }} if py_limited_api else {},
)
