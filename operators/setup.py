from setuptools import setup, Extension
import numpy

module = Extension('rknn_infer_dfine',
                   sources=['gridsample_op_cpu.cpp'], # Your C source file
                   include_dirs=['/home/radxa/PythonWorkspace/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include', numpy.get_include()], # Include paths
                   library_dirs=['/home/radxa/PythonWorkspace/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64'], # Library path
                   libraries=['rknnrt'], # Link against librknnrt
                   extra_compile_args=['-O3'])

setup(name='dfineRKNNInfer',
      version='1.0',
      description='RKNN inference using C extension',
      ext_modules=[module])