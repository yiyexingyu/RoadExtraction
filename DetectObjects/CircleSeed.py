# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:06
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : CircleSeed.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import sqrt
import cv2
import numpy as np
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor, QPainterPath, QImage
from .Utils import GrayPixelsSet, calculate_reference_color_of, get_pixels_from, calculate_spectral_distance, bound
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters
from Core.PeripheralCondition import PeripheralCondition
from Core.DetectionStrategy.Strategy import DetectionStrategy
from Core.GrayLevelCooccurrenceMatrix import GrayLCM, get_circle_spectral_vector


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
        self._parent_seed = parent_seed

        self._peripheral_condition = None  # type: PeripheralCondition
        self._spectral_distance = 255
        self._spectral_info_vector = self._calculate_spectral_info_vector()
        self._calculate_spectral_distance()

        self._general_strategy = DetectionStrategy.Initialization
        self._child_seeds = []

    def construct_road_path(self, detection_param: DetectionParameters):
        """构建圆形种子对应的道路的路径"""

    def intersected_road_path_with(self, other):
        """和其他种子中道路部分的交集"""

    def _calculate_texture_info_vector(self):
        """计算种子的纹理特征"""
        if not self._seed_pixels:
            return None

        # 计算对比度

    def _calculate_spectral_info_vector(self):
        """计算种子的光谱特征"""
        if not self._seed_pixels:
            return None
        sr, sg, sb = 0, 0, 0
        for seed_pixel in self._seed_pixels:
            sr += seed_pixel.r()
            sg += seed_pixel.g()
            sb += seed_pixel.b()

        num_pixels = len(self._seed_pixels)
        xr = sr / num_pixels
        xg = sg / num_pixels
        xb = sb / num_pixels

        sr, sg, sb = 0, 0, 0
        for seed_pixel in self._seed_pixels:
            sr += (seed_pixel.r() - xr) ** 2
            sg += (seed_pixel.g() - xg) ** 2
            sb += (seed_pixel.b() - xb) ** 2
        sdr = sqrt(sr / num_pixels)
        sdg = sqrt(sg / num_pixels)
        sdb = sqrt(sb / num_pixels)

        return [xr, xg, xb, sdr, sdg, sdb]

    def _calculate_spectral_distance(self):
        if self._parent_seed is not None:
            self._spectral_distance = calculate_spectral_distance(
                self._spectral_info_vector, self._parent_seed.spectral_info_vector)

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
        general_strategy = "general_strategy: " + self.general_strategy.value + "\n"
        return position_info + radius_info + direction_info + re_color_info + general_strategy

    def init_circle_seed(self, image: QImage):
        self._seed_pixels = get_pixels_from(image, self._position, self._radius)
        self._gray_seed_pixels = GrayPixelsSet.get_gray_pixels_from(self._seed_pixels)
        self._reference_color = calculate_reference_color_of(self)
        self._spectral_info_vector = self._calculate_spectral_info_vector()
        self._calculate_spectral_distance()

    def set_position(self, position: QPoint):
        self._position = position

    @property
    def peripheral_condition(self) -> PeripheralCondition:
        return self._peripheral_condition

    @peripheral_condition.setter
    def peripheral_condition(self, condition):
        if isinstance(condition, PeripheralCondition):
            self._peripheral_condition = condition

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

    def set_radius(self, radius: int):
        self._radius = radius

    @property
    def spectral_distance(self):
        return self._spectral_distance

    def __lt__(self, other):
        return self._spectral_distance < other.spectral_distance

    @property
    def spectral_info_vector(self):
        return self._spectral_info_vector

    @property
    def general_strategy(self) -> DetectionStrategy:
        return self._general_strategy

    @general_strategy.setter
    def general_strategy(self, strategy: DetectionStrategy):
        self._general_strategy = strategy

    @property
    def position(self) -> QPoint:
        return self._position

    def set_position(self, image: QImage, new_position: QPoint):
        self._position = new_position
        self.init_circle_seed(image)

    @property
    def radius(self) -> int:
        return self._radius

    def set_radius(self, image: QImage, new_radius: int):
        self._radius = new_radius
        self.init_circle_seed(image)

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


class CircleSeedNp:
    """
    基于numpy数组的圆形探测种子，检测跟踪道路的主对象。这个类定义了一个圆形种子（circle seed)的所有必要属性和行为
        :param position: 圆形种子圆心相对于源图片左上角的位置
        :type position: list
        :param radius: 圆形种子的半径
        :type radius: int
        :param generate_strategy: 种子产生的策略
        :type generate_strategy: DetectionStrategy
        :param image: 源图片的np矩阵，RGB格式
        :type image: np.ndarray
        :param direction: 圆形种子的方向，范围：[0, 2*pi)
        :type direction: float
        :param parent_seed: 圆形种子的方向
        :type CircleSeedNp
    """

    def __init__(self, position, radius, generate_strategy, image=None, direction=0, parent_seed=None):
        """
        基于numpy数组的圆形探测种子，检测跟踪道路的主对象
        :param position: 圆形种子圆心相对于源图片左上角的位置
        :type position: list
        :param radius: 圆形种子的半径
        :type radius: int
        :param generate_strategy: 种子产生的策略
        :type generate_strategy: DetectionStrategy
        :param image: 源图片的np矩阵，RGB格式
        :type image: np.ndarray
        :param direction: 圆形种子的方向，范围：[0, 2*pi)
        :type direction: float
        :param parent_seed: 圆形种子的方向
        :type CircleSeedNp
        """
        if isinstance(position, QPoint):
            position = [position.x(), position.y()]
        # 圆形种子四元素
        self._position = position
        self._radius = radius
        self._direction = direction
        self._seed_pixels = np.zeros((radius * 2, radius * 2, 3))

        # 圆形种子附加信息
        self._parent_seed = parent_seed
        self._child_seeds = []
        self._generate_strategy = generate_strategy

        # 圆形种子的特征信息: 光谱特征向量和纹理特征向量
        self._spectral_feature_vector = []
        self._texture_feature_vector = []

        if isinstance(image, np.ndarray):
            self.initialization(image)

    def initialization(self, image: np.ndarray):
        assert image.ndim >= 3
        # 从源图片中截取出圆形种子范围内的像素
        # 裁剪坐标为[y0:y1, x0:x1], ------- 这里要进行边界处理： <0 > width, height的情况
        y0 = bound(0, self._position[1] - self._radius, image.shape[0])
        y1 = bound(0, self._position[1] + self._radius, image.shape[0])
        x0 = bound(0, self._position[0] - self._radius, image.shape[1])
        x1 = bound(0, self._position[0] + self._radius, image.shape[1])
        self._seed_pixels = image[y0: y1, x0: x1, :]   # type: np.ndarray

        seed_pixels_gray = cv2.cvtColor(self._seed_pixels, cv2.COLOR_RGB2GRAY).astype(np.int16)
        self._seed_pixels = self._seed_pixels.astype(np.int16)
        # 圆形提取模板矩阵
        temp_img = -np.ones((self._seed_pixels.shape[0], self._seed_pixels.shape[1], 1)) * 256
        temp_img = cv2.circle(temp_img, (self.radius - 1, self.radius - 1), self._radius, 0, -1).astype(np.int16)
        # 进行提取 为了避免有些值为0的像素可用-256的矩阵进行相加
        self._seed_pixels[:, :, 0] = np.add(self._seed_pixels[:, :, 0], temp_img[:, :, 0])
        self._seed_pixels[:, :, 1] = np.add(self._seed_pixels[:, :, 1], temp_img[:, :, 0])
        self._seed_pixels[:, :, 2] = np.add(self._seed_pixels[:, :, 2], temp_img[:, :, 0])

        seed_pixels_gray += temp_img[:, :, 0]
        # self._seed_pixels[self._seed_pixels < 0] = -1

        # 计算种子的特征信息
        self._spectral_feature_vector = get_circle_spectral_vector(self._seed_pixels)
        # self._texture_feature_vector = GrayLCM.get_road_texture_vector(seed_pixels_gray)

    def set_position(self, position: list, image: np.ndarray):
        self._position = position
        self.initialization(image)

    def set_radius(self, radius: int, image: np.ndarray):
        self._radius = radius
        self.initialization(image)

    def get_pixels_position(self, image_size: tuple) -> np.ndarray:
        x_axis = np.arange(self._position[0] - self._radius, self._position[0] + self._radius, 1, np.int)
        y_axis = np.arange(self._position[1] - self._radius, self._position[1] + self._radius, 1, np.int)
        pixels_position = np.array(np.meshgrid(*[x_axis, y_axis])).T

        temp_mod = self._seed_pixels[:, :, 0]
        pixels_position = pixels_position[temp_mod >= 0]

        # pixels_position = pixels_position[temp_mod >= 0].astype('str')
        # pixels_position = np.char.add(np.char.add(pixels_position[:, 0], ','), pixels_position[:, 1])
        # pixels_position = pixels_position[:, 1] * image_size[1] + pixels_position[:, 0]
        return pixels_position

    def append_child_seed(self, child_seed):
        self._child_seeds.append(child_seed)

    def children(self):
        return self._child_seeds

    def __str__(self):
        position_info = "position(" + str(self._position[0]) + "," + str(self._position[1]) + ")\n"
        radius_info = "radius=" + str(self._radius) + "\n"
        direction_info = "direction=" + str(self._direction) + "\n"
        general_strategy = "generate_strategy: " + str(self._generate_strategy.value) + "\n"
        spectral_vector = "spectral_vector: " + str(self._spectral_feature_vector) + "\n"
        texture_vector = "texture_vector: " + str(self._texture_feature_vector) + "\n"
        spectral_distance = "spectral_distance: " + str(self.spectral_distance) + "\n"
        texture_distance = "texture_distance: " + str(self.texture_distance) + "\n"
        child_seeds_len = "子种子个数: " + str(len(self._child_seeds)) + "\n"
        return position_info + radius_info + direction_info + general_strategy + spectral_distance + texture_distance \
               + spectral_vector + texture_vector + child_seeds_len

    @property
    def spectral_distance(self):
        if self._parent_seed:
            return np.linalg.norm(np.subtract(self._spectral_feature_vector, self._parent_seed.spectral_feature_vector))
        return 0.

    @property
    def texture_distance(self):
        if self._parent_seed:
            return np.linalg.norm(np.subtract(self._texture_feature_vector, self._parent_seed.texture_feature_vector))
        return 0.

    @property
    def position(self):
        return self._position

    @property
    def radius(self):
        return self._radius

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, new_direction):
        self._direction = new_direction

    @property
    def seed_pixels(self) -> np.ndarray:
        return self._seed_pixels

    @property
    def parent_seed(self):
        return self._parent_seed

    @parent_seed.setter
    def parent_seed(self, parent):
        self._parent_seed = parent

    @property
    def spectral_feature_vector(self):
        return self._spectral_feature_vector

    @property
    def texture_feature_vector(self):
        return self._texture_feature_vector

    @property
    def generate_strategy(self) -> DetectionStrategy:
        return self._generate_strategy

    @generate_strategy.setter
    def generate_strategy(self, strategy: DetectionStrategy):
        if isinstance(strategy, DetectionStrategy):
            self._generate_strategy = strategy

    def __lt__(self, other):
        res = self.spectral_distance < other.spectral_distance
        return res
