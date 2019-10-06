# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 17:12
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : RoadDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint
from .Exception import NoInitializeError


class RoadDetection:

    def __init__(self, image: QImage):
        self._image = image
        self._seed_list = []
        self._middle_distance = 12
        # self._angle_interval = [1, pi, 12]
        self._angle_interval = pi / 12

    def initialize(self, position: QPoint, radius: int):
        """
        :param position:
        :param radius:
        :return:
        """

    def road_detection(self):
        if len(self._seed_list) <= 0:
            raise NoInitializeError()

    def _generate_candidate_seed(self):
        """TODO"""

    def calculate_peripheral_condition_of(self, circle_seed):
        """TODO"""

    def _validation_adjacent_seeds(self):
        """TODO"""
