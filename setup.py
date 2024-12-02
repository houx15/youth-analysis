from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("extract_user_id.pyx", language_level="3"),
)
