from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("user_id_counter.pyx"),
)
