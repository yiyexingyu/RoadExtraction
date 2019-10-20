# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 20:08
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : SDDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5.QtGui import QImage
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters
from Core.DetectionAlgorithm.SingleDirectionDetection import single_direction_detection
from DetectObjects.CircleSeed import CircleSeed
from .AbstractDetectionStrategy import AbstractDetectionStrategy


class SDDetectionStrategy(AbstractDetectionStrategy):

    def __init__(self):
        super(SDDetectionStrategy, self).__init__()

    def road_detect(self, image: QImage, parent_seed: CircleSeed, detection_param: DetectionParameters, angle_interval):
        return single_direction_detection(image, parent_seed, detection_param, angle_interval)
