from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("extract_user_id_special.pyx"),
)
