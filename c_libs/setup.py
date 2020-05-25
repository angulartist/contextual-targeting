from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Cython',
    ext_modules=cythonize(['core/*.pyx']),
)
