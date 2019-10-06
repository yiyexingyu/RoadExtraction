# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 13:07
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : Pixels.py
# @Project : RoadExtraction
# @Software: PyCharm
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor


class SeedPixel:

    def __init__(self, position: QPoint, color: QColor):
        self._position = position
        self._color = color

    @property
    def position(self):
        return self._position

    @property
    def color(self):
        return self._color

    def x(self):
        return self._position.x()

    def y(self):
        return self._position.y()

    def r(self):
        return self._color.red()

    def g(self):
        return self._color.green()

    def b(self):
        return self._color.blue()


class SeedPixelsSet:

    def __init__(self):
        """"""
