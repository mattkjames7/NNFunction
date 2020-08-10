import setuptools
from setuptools.command.install import install
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NNFunction",
    version="0.0.1",
    author="Matthew Knight James",
    author_email="mattkjames7@gmail.com",
    description="A simple package for modelling multidimensional non-linear functions using artificial neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattkjames7/NNFunction",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: POSIX",
    ],
    install_requires=[
		'numpy',
		'PyFileIO',
		'matplotlib',
		'tensorflow',
		'scikit-learn',
	],
	include_package_data=True,
)



