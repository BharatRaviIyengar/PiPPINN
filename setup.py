from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension #, CUDAExtension, CUDA_HOME

def get_extension():
	return CppExtension(
			'NeighborhoodRestriction',  # name of the extension
			sources=['Neighborhood_restriction.cpp'],
			extra_compile_args=['-O3', '-fopenmp'],
			extra_link_args=['-fopenmp']
		)

setup(
	name='NeighborhoodRestriction',
	ext_modules=[get_extension()],
	cmdclass={'build_ext': BuildExtension}
)