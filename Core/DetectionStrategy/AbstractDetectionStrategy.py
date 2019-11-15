# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 18:16
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : AbstractDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

import time
import numpy as np
from numpy import ndarray
from math import sin, cos
from abc import abstractmethod
from DetectObjects.CircleSeed import CircleSeedNp
from DetectObjects.Utils import adjust_angle
from Core.DetectionStrategy.Strategy import DetectionStrategy


class AbstractDetectionStrategy:

    def __init__(self):
        """TODO"""

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
                                          image, current_angle, parent_seed=parent_seed, ref_spectral=ref_seed)
            return candidate_seed
        if detection_param.ERA is None:
            return [create_candidate_seed(parent_seed.direction)]

        moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_seed.radius
        candidate_seeds = []

        # half_era = detection_param.ERA / 2 + angle_interval / 2
        # # 使用numpy批量生成种子
        # # 首先批量生成方向
        # half_era_angle = np.arange(0, half_era, angle_interval)
        # directions = np.union1d(adjust_angle(half_era_angle + parent_seed.direction),
        #                         adjust_angle(parent_seed.direction - half_era_angle))
        #
        # # 然后批量生成坐标
        # x_candidates = parent_seed.position[0] + (moving_distance * np.sin(directions)).astype(np.int32)
        # y_candidates = parent_seed.position[1] - (moving_distance * np.cos(directions)).astype(np.int32)
        #
        # # 进行坐标筛选
        # x_bound = (x_candidates >= parent_seed.radius) & (x_candidates <= image.shape[1] - parent_seed.radius)
        # x_candidates = x_candidates[x_bound]
        # y_candidates = y_candidates[x_bound]
        # directions = directions[x_bound]
        #
        # y_bound = (y_candidates >= parent_seed.radius) & (y_candidates <= image.shape[0] - parent_seed.radius)
        # x_candidates = x_candidates[y_bound]
        # y_candidates = y_candidates[y_bound]
        # directions = directions[y_bound]
        #
        # # 创建候选种子
        # t = time.time()
        # if directions.size <= 0:
        #     return []
        #
        # candidate_seeds = [
        #     CircleSeedNp([int(x), int(y)], parent_seed.radius, detection_strategy, image, float(direction), ref_seed)
        #     for x, y, direction in np.nditer([x_candidates, y_candidates, directions])
        # ]
        # return candidate_seeds

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
