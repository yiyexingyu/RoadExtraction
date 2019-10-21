# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:06
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : CircleSeed.py
# @Project : RoadExtraction
# @Software: PyCharm
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor, QPainterPath, QImage
from .Utils import GrayPixelsSet, calculate_reference_color_of, get_pixels_from
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters


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

    def __init__(self, position: QPoint, radius: int, seed_pixels: [], direction: float = 0., parent_seed=None):
        self._position = position
        self._radius = radius
        self._direction = direction
        self._seed_pixels = seed_pixels
        self._road_path = QPainterPath()
        self._gray_seed_pixels = GrayPixelsSet.get_gray_pixels_from(seed_pixels)
        self._reference_color = calculate_reference_color_of(self)

        self._general_strategy = "init"
        self._parent_seed = parent_seed
        self._child_seeds = []

    def construct_road_path(self, detection_param: DetectionParameters):
        """构建圆形种子对应的道路的路径"""

    def intersected_road_path_with(self, other):
        """和其他种子中道路部分的交集"""

    def intersected_with(self, road_seed):
        road_pixels = road_seed.seed_pixels
        result = list(set(self._seed_pixels) & set(road_pixels))
        return result

    def intersected_pixel_num_with(self, road_seed):
        return len(self.intersected_with(road_seed))

    def append_child_seed(self, child_seed):
        if isinstance(child_seed, CircleSeed):
            self._child_seeds.append(child_seed)

    def extend_child_seeds(self, child_seeds: []):
        for child_seed in child_seeds:
            if not isinstance(child_seed, CircleSeed):
                continue
            self._child_seeds.append(child_seed)

    def __str__(self):
        position_info = "position(" + str(self._position.x()) + "," + str(self._position.y()) + ")\n"
        radius_info = "radius=" + str(self._radius) + "\n"
        direction_info = "direction=" + str(self._direction) + "\n"
        re_color_info = "reference color(" + str(self.reference_color.red()) + "," + str(
            self.reference_color.green()) + "," + str(self._reference_color.blue()) + ")\n"
        return position_info + radius_info + direction_info + re_color_info

    def init_circle_seed(self, image: QImage):
        self._seed_pixels = get_pixels_from(image, self._position, self._radius)
        self._gray_seed_pixels = GrayPixelsSet.get_gray_pixels_from(self._seed_pixels)
        self._reference_color = calculate_reference_color_of(self)

    @property
    def child_seeds(self) -> ():
        return tuple(self._child_seeds)

    @property
    def parent_seed(self):
        return self._parent_seed

    @parent_seed.setter
    def parent_seed(self, new_parent_seed):
        if not isinstance(new_parent_seed, CircleSeed):
            return
        self._parent_seed = new_parent_seed

    def set_position(self, position: QPoint):
        self._position = position

    def set_radius(self, radius: int):
        self._radius = radius

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

    @property
    def gray_pixels(self):
        return self._gray_seed_pixels
