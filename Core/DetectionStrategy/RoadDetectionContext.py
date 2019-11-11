# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 19:19
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : RoadDetectionContext.py
# @Project : RoadExtraction
# @Software: PyCharm

from .AbstractDetectionStrategy import AbstractDetectionStrategy


class RoadDetectionContext:

    def __init__(self, road_detection_strategy: AbstractDetectionStrategy = None):
        if isinstance(road_detection_strategy, AbstractDetectionStrategy):
            self._road_detection_strategy = road_detection_strategy
        else:
            self._road_detection_strategy = None

    def road_detection(self, image, parent_seed, ref_seed, angle_interval, detection_strategy, detection_param):
        if not isinstance(self._road_detection_strategy, AbstractDetectionStrategy):
            return []
        return self.road_detection_strategy.road_detection(
            image, parent_seed, ref_seed, angle_interval, detection_strategy, detection_param)

    @property
    def road_detection_strategy(self) -> AbstractDetectionStrategy:
        return self._road_detection_strategy

    @road_detection_strategy.setter
    def road_detection_strategy(self, strategy: AbstractDetectionStrategy):
        if not isinstance(strategy, AbstractDetectionStrategy):
            return
        self._road_detection_strategy = strategy
