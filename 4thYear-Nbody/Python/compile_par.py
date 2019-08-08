from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension(
                        'Gravity_par', ['Gravity_par.pyx'],
                        extra_compile_args=["-fopenmp","-ffast-math"],
                        extra_link_args=["-fopenmp"],
                        )
               ]

setup(name='Gravity_par',ext_modules=cythonize(ext_modules))
