import setuptools
# -*- encoding: utf-8 -*-
# @File    : setup.py
# @Time    : 2019/11/22 20:12
# @Author  : 一叶星羽
# @Email   : h0670131005@gmail.com
# @Software: PyCharm
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("test.pyx")
)
