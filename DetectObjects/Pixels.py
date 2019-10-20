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

    def __eq__(self, other):
        return self._position == other.position

    def __hash__(self):
        return hash(str(self._position.x()) + str(self._position.y()))

    def __str__(self):
        return "pixel(" + str(self._position.x()) + "," + str(self.position.y()) + "," + self.color.name() + ")"

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


if __name__ == '__main__':
    pixels_set1 = {SeedPixel(QPoint(12, 23), QColor(2)),
                   SeedPixel(QPoint(21, 24), QColor(2)),
                   SeedPixel(QPoint(55, 46), QColor(2)),
                   SeedPixel(QPoint(33, 23), QColor(2))}
    pixels_set2 = {SeedPixel(QPoint(12, 23), QColor(2)),
                   SeedPixel(QPoint(21, 25), QColor(2)),
                   SeedPixel(QPoint(58, 46), QColor(2)),
                   SeedPixel(QPoint(33, 23), QColor(2))}
    print(list(pixels_set1 & pixels_set2))
    for pixel in pixels_set1 & pixels_set2:
        print(pixel)
