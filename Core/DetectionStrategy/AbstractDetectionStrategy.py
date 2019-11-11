# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 18:16
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : AbstractDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

import numpy as np
from numpy import ndarray
from math import sin, cos
from abc import abstractmethod
from PyQt5.QtGui import QImage
from DetectObjects.CircleSeed import CircleSeedNp
from DetectObjects.Utils import adjust_angle
# from ..DetectionAlgorithm.DetectionParameters import DetectionParameters


class AbstractDetectionStrategy:

    def __init__(self):
        """TODO"""

    @abstractmethod
    def road_detect(self, image: QImage, parent_seed, detection_param, angle_interval):
        raise NotImplementedError

    @abstractmethod
    def analysis_peripheral_condition(self, candidate_seeds: list, detection_param):
        raise NotImplementedError

    @abstractmethod
    def road_detection(self, image, ref_seed, parent_seed, angle_interval, detection_strategy, detection_param):
        raise NotImplementedError

    @staticmethod
    def generate_candidate_seeds(image, parent_seed, ref_seed, angle_interval, detection_strategy, detection_param):
        """
        根据父种子、检测策略和有效角度范围生成候选种子（未经筛选）。
        当不指定有效有效角度范围时，会沿着父种子的方向生成一个候选种子
        :param image: 源图片像素矩阵：rgb格式
        :type image: ndarray
        :param parent_seed: 用于生成候选种子的父种子
        :type parent_seed: CircleSeed
        :param ref_seed: 光谱信息参考种子
        :type ref_seed: CircleSeedNp
        :param angle_interval: 角度间距
        :type angle_interval: float
        :param detection_strategy: 道路跟踪检测的策略
        :type detection_strategy: DetectionStrategy
        :param detection_param: 道路跟踪检测的常数参数
        :type detection_param: DetectionParameters
        :return: 返回生成的候选种子列表
        :rtype: list
        """

        def create_candidate_seed(current_angle: float) -> [CircleSeedNp, None]:
            x_candidate = parent_seed.position[0] + int(moving_distance * cos(current_angle))
            y_candidate = parent_seed.position[1] - int(moving_distance * sin(current_angle))

            bound_x = (x_candidate - parent_seed.radius < 0, x_candidate + parent_seed.radius > image.shape[1])
            bound_y = (y_candidate - parent_seed.radius < 0, y_candidate + parent_seed.radius > image.shape[0])
            if any(bound_x) or any(bound_y):
                return None
            current_pos = [x_candidate, y_candidate]

            # 创建候选种子
            candidate_seed = CircleSeedNp(current_pos, parent_seed.radius, detection_strategy,
                                          image, current_angle, parent_seed=ref_seed)
            return candidate_seed
        if detection_param.ERA is None:
            return [create_candidate_seed(parent_seed.direction)]
        moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_seed.radius
        candidate_seeds = []
        half_era = detection_param.ERA / 2

        for k in range(0, int(half_era / angle_interval) + 1):
            pos_current_angle = adjust_angle(k * angle_interval + parent_seed.direction)
            nes_current_angle = adjust_angle(parent_seed.direction - k * angle_interval)

            circle_seed = create_candidate_seed(pos_current_angle)
            if circle_seed:
                candidate_seeds.append(circle_seed)
            if k != 0:
                circle_seed = create_candidate_seed(nes_current_angle)
                if circle_seed:
                    candidate_seeds.append(circle_seed)
        return candidate_seeds
