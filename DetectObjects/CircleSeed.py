# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:06
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : CircleSeed.py
# @Project : RoadExtraction
# @Software: PyCharm
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor
from .Utils import GrayPixelsSet, calculate_reference_color_from
from .Pixels import SeedPixel


class CircleSeed:
    """
    圆形探测种子, 检测跟踪道路的主要对象。
    _position 位置相对于图片的左上角的位置;
    _radius 种子的半径, 以像素为单位;
    _direction 种子(探测)的方向， 范围0-2π， -1表示是初始种子;
    _reference_color 圆形种子的参考色
    _seed_pixels 圆形种子覆盖的像素
    _gray_seed_pixels 圆形种子覆盖的灰色像素，是_seed_pixels和灰色像素集的交集
    """

    def __init__(self, position: QPoint, radius: int, seed_pixels: [], direction: float = -1.):
        self._position = position
        self._radius = radius
        self._direction = direction
        self._seed_pixels = seed_pixels
        self._gray_seed_pixels = GrayPixelsSet.get_gray_pixels_from(seed_pixels)
        self._reference_color = calculate_reference_color_from(self._gray_seed_pixels)

    @property
    def position(self) -> QPoint:
        return self._position

    @position.setter
    def position(self, new_position: QPoint):
        self._position = new_position

    @property
    def radius(self) -> int:
        return self._radius

    @radius.setter
    def radius(self, new_radius: int):
        self._radius = new_radius

    @property
    def direction(self) -> float:
        return self._direction

    @direction.setter
    def direction(self, new_direction: int):
        self._direction = new_direction

    @property
    def reference_color(self) -> QColor:
        return self._reference_color

    @property
    def seed_pixels(self) -> tuple:
        return tuple(self._seed_pixels)
