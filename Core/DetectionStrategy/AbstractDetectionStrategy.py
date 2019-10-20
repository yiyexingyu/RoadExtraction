# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 18:16
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : AbstractDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi
from abc import abstractmethod
from PyQt5.QtGui import QImage
from DetectObjects.CircleSeed import CircleSeed
from ..DetectionAlgorithm.DetectionParameters import DetectionParameters


class AbstractDetectionStrategy:

    def __init__(self):
        """TODO"""

    @abstractmethod
    def road_detect(self, image: QImage, parent_seed: CircleSeed, detection_param: DetectionParameters, angle_interval):
        raise NotImplementedError
