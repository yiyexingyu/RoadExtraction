# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 17:12
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : RoadDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi, cos, sin
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint
from .Exception import NoInitializeError
from .DetectionAlgorithm.DetectionParameters import DetectionParameters
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import get_pixels_from
from .DetectionAlgorithm.GeneralSimilarityDetection import general_similarity_detection
from .DetectionAlgorithm.GraySimilarityDetection import gray_similarity_detection
from .DetectionAlgorithm.JumpSimilarityDetection import jump_similarity_detection
from .DetectionAlgorithm.NarrowSimilarityDetection import narrow_similarity_detection
from .DetectionAlgorithm.SingleDirectionDetection import single_direction_detection


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
        circle_seed_pixels = get_pixels_from(self._image, position, radius)
        circle_seed = CircleSeed(position, radius, circle_seed_pixels, direction=-2 * pi)
        self._seed_list.append(circle_seed)

    def road_detection(self):
        if len(self._seed_list) <= 0:
            raise NoInitializeError()

    def _generate_candidate_seed(self, parent_circle_seed: CircleSeed):
        detection_param = DetectionParameters.generate_multi_directional_general_similarity_detection_parameters(
                parent_circle_seed.radius)
        moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_circle_seed.radius

        candidate_seeds = []
        for k in range(1, int(2 * pi / self._angle_interval)):
            current_angle = k * self._angle_interval
            x_candidate = parent_circle_seed.position.x() + int(moving_distance * cos(current_angle))
            y_candidate = parent_circle_seed.position.y() + int(moving_distance * sin(current_angle))
            current_pos = QPoint(x_candidate, y_candidate)

            seed_pixels = get_pixels_from(self._image, current_pos, parent_circle_seed.radius)
            candidate_seed = CircleSeed(current_pos, parent_circle_seed.radius, seed_pixels, current_angle)

            candidate_seeds.append(candidate_seed)

    def calculate_peripheral_condition_of(self, circle_seed):
        """TODO"""

    def _validation_adjacent_seeds(self):
        """TODO"""
