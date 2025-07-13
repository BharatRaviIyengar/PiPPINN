from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name='grouped_multinomial',
	ext_modules=[
		CUDAExtension(
			name='grouped_multinomial',
			sources=['grouped_multinomial.cpp', 'grouped_multinomial_kernel.cu'],
			extra_compile_args={
				'cxx': ['-O3'],
				'nvcc': [
					'-O3',
					'-use_fast_math',
					'-DMAX_K=32',
					'-gencode=arch=compute_80,code=sm_80',  # A100
					'-gencode=arch=compute_86,code=sm_86',  # RTX 3090
					'-gencode=arch=compute_89,code=sm_89',  # RTX 4090
					'-gencode=arch=compute_89,code=compute_89'  # PTX for forward compatibility
				]
			}
		)
	],
	cmdclass={'build_ext': BuildExtension}
)
