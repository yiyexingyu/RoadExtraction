# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 16:47
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : JumpSimilarityDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi, cos, sin
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import get_pixels_from
from .DetectionParameters import DetectionParameters
from .GeneralSimilarityDetection import general_similarity_detection
from ..PeripheralCondition import calculate_peripheral_condition


def jump_similarity_detection(
        image: QImage, parent_circle_seed: CircleSeed, detection_param: DetectionParameters, angle_interval=pi / 12):
    """
    :param image: 源遥感图片
    :param parent_circle_seed: 父圆形种子
    :param detection_param: 检测算法所需参数
    :param angle_interval:  角度间距
    :return: 有效的候选种子数组
    """

    moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_circle_seed.radius + detection_param.DI
    candidate_seeds, last_road_pixels_proportion = None, 0
    general_detection_param = DetectionParameters.generate_multi_directional_general_similarity_detection_parameters(
        parent_circle_seed.radius)
    general_detection_param.MD = moving_distance
    general_detection_param.CDiff = detection_param.CDiff

    while moving_distance <= detection_param.MMD:
        candidate_seeds = general_similarity_detection(
            image, parent_circle_seed, general_detection_param, angle_interval)

        if candidate_seeds:
            break
        moving_distance += detection_param.DI
        general_detection_param.MD = moving_distance

    return candidate_seeds
