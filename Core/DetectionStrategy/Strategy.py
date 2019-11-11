# -*- coding: utf-8 -*-
# @Time    : 2019/11/9 23:36
# @Author  : 一叶星羽
# @Email   : 2958029539@qq.com
# @File    : Strategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from enum import Enum


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
