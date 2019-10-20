# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 18:27
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : GNSDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5.QtGui import QImage
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters
from Core.DetectionAlgorithm.GeneralSimilarityDetection import general_similarity_detection
from DetectObjects.CircleSeed import CircleSeed
from .AbstractDetectionStrategy import AbstractDetectionStrategy


class GNSDetectionStrategy(AbstractDetectionStrategy):

    def __init__(self):
        super(GNSDetectionStrategy, self).__init__()

    def road_detect(self, image: QImage, parent_seed: CircleSeed, detection_param: DetectionParameters, angle_interval):
        return general_similarity_detection(image, parent_seed, detection_param, angle_interval)
