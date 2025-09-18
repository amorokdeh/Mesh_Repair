from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'QEM',
        ['QEM.cpp'],
        include_dirs=[
            pybind11.get_include(),
            r'C:\Libraries\Eigen-3.4.0'
        ],
        language='c++'
    ),
    Extension(
        'curvature_simplification',
        ['curvature_simplification.cpp'],  # updated here
        include_dirs=[
            pybind11.get_include(),
            r'C:\Libraries\Eigen-3.4.0'
        ],
        language='c++',
    )
]

setup(
    name='mesh_extensions',
    version='0.1',
    ext_modules=ext_modules,
)
