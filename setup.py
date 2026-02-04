from setuptools import setup, Extension
import numpy
import os

# Define the extension module
mpyfit_ext = Extension(
    name='mpyfit.mpfit',
    sources=[
        'mpyfit/mpyfit.c',
        'mpyfit/cmpfit/mpfit.c'
    ],
    include_dirs=[
        numpy.get_include(),
        'mpyfit/cmpfit'
    ]
)

setup(
    name='mpyfit',
    version='1.0',
    description='A wrapper around C mpfit',
    packages=['mpyfit'],
    ext_modules=[mpyfit_ext],
)
