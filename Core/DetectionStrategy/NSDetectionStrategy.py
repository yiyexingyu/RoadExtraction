# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 20:06
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : NSDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5.QtGui import QImage
from .AbstractDetectionStrategy import AbstractDetectionStrategy


class NSDetectionStrategy(AbstractDetectionStrategy):

    def __init__(self):
        super(NSDetectionStrategy, self).__init__()

    def road_detection(self, image, ref_seed, parent_seed, angle_interval, detection_strategy, detection_param):
        """"""
