from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension(
                        'Gravity_lin', ['Gravity_lin.pyx'],
                        extra_compile_args=["-ffast-math","-fopenmp"],
                        extra_link_args=["-fopenmp"],
                        )
               ]

setup(name='Gravity_lin',ext_modules=cythonize(ext_modules))
