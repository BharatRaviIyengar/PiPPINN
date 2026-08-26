from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent


setup(
	name="pippinn-neighborhood-restriction",
	version="0.1.0",
	ext_modules=[
		CppExtension(
			name="NeighborhoodRestriction",
			sources=[str(ROOT / "Neighborhood_restriction.cpp")],
			extra_compile_args=["-O3", "-std=c++17", "-fopenmp"],
			extra_link_args=["-fopenmp"],
		)
	],
	cmdclass={"build_ext": BuildExtension},
)
