from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
	name='batch_multinomial',
	ext_modules=[
		CppExtension(
			'batch_multinomial',
			['multinomial_loop.cpp'],  # or include CUDA files here if needed
			extra_compile_args=['-O3','-fopenmp'],
			extra_link_args=['-fopenmp']
		)
	],
	cmdclass={
		'build_ext': BuildExtension
	}
)
