from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

def get_extension():
	if CUDA_HOME is not None:
		return CUDAExtension(
			'cpputils',
			sources=['cpputils.cpp', 'remap_cuda.cu'],
			extra_compile_args={
				'cxx': ['-O3','-fopenmp','-DWITH_CUDA'],
				'nvcc': ['-O3','-DWITH_CUDA']
			},
			extra_link_args=['-fopenmp']
		
		)
	else:
		return CppExtension(
			'cpputils',
			sources=['cpputils.cpp'],
			extra_compile_args=['-O3', '-fopenmp'],
			extra_link_args=['-fopenmp']
		)

setup(
	name='cpputils',
	ext_modules=[get_extension()],
	cmdclass={'build_ext': BuildExtension}
)