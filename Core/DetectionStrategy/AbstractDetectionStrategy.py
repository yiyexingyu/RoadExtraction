# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 18:16
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : AbstractDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from enum import Enum
from abc import abstractmethod
from PyQt5.QtGui import QImage
# from DetectObjects.CircleSeed import CircleSeed
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


class DetectionStrategy(Enum):

    Initialization = "Initialization"
    MultiGNSDetectionStrategy = "MultiGNSDetectionStrategy"
    MultiGRSDetectionStrategy = "MultiGRSDetectionStrategy"
    MultiJSDetectionStrategy = "MultiJSDetectionStrategy"
    MultiNSDetectionStrategy = "MultiNSDetectionStrategy"

    SingleGRSDetectionStrategy = "SingleGRSDetectionStrategy"
    SingleJSDetectionStrategy = "SingleJSDetectionStrategy"
    SingleNSDetectionStrategy = "SingleNSDetectionStrategy"
    SingleSSDetectionStrategy = "SingleSSDetectionStrategy"
