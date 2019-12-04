# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:06
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : CircleSeed.py
# @Project : RoadExtraction
# @Software: PyCharm

import cv2
import numpy as np
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPainterPath
from .Utils import bound
from Core.DetectionStrategy.Strategy import DetectionStrategy
from Core.GrayLevelCooccurrenceMatrix import get_circle_spectral_vector, road_texture_vector


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

    def __init__(self, position, radius, generate_strategy, image=None, direction=0, parent_seed=None, ref_spectral=None):
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
        self._spectral_distance = 0.

        if isinstance(image, np.ndarray):
            self.initialization(image)
            if self._generate_strategy != DetectionStrategy.Initialization:
                self._spectral_distance = np.linalg.norm(np.subtract(self._spectral_feature_vector, ref_spectral))

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
        self._texture_feature_vector = road_texture_vector(seed_pixels_gray)

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
    def road_path(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(QPoint(self._position[0], self._position[1]), self._radius, self._radius)
        return path

    @property
    def road_feature_vector(self):
        return np.hstack((self._texture_feature_vector, self._spectral_feature_vector))

    @property
    def road_feature_distance(self):
        if self._parent_seed:
            return np.linalg.norm(np.subtract(self.road_feature_vector, self._parent_seed.road_feature_vector))
        return 0.

    @property
    def spectral_distance(self):
        return self._spectral_distance

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
