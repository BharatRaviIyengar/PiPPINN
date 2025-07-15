from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
	name='max_nbr',
	ext_modules=[
		CppExtension(
			'max_nbr',
			['max_nbr.cpp'],  # or include CUDA files here if needed
			extra_compile_args=['-O3','-fopenmp'],
			extra_link_args=['-fopenmp']
		)
	],
	cmdclass={
		'build_ext': BuildExtension
	}
)
