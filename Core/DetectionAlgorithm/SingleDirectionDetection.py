# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 16:51
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : SingleDirectionDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi, cos, sin
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import get_pixels_from
from .DetectionParameters import DetectionParameters
from ..PeripheralCondition import calculate_peripheral_condition


def single_direction_detection(
        image: QImage, parent_circle_seed: CircleSeed, detection_param: DetectionParameters, angle_interval=pi / 12):
    """
    :param image: 源遥感图片
    :param parent_circle_seed: 父圆形种子
    :param detection_param: 检测算法所需参数
    :param angle_interval:  角度间距
    :return: 有效的候选种子数组
    """

    if parent_circle_seed.general_strategy == "init":
        return None

    moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_circle_seed.radius
    candidate_seeds, similarity_gray_pixels_proportion = None, 0

    current_angle = parent_circle_seed.direction

    x_candidate = parent_circle_seed.position.x() + int(moving_distance * cos(current_angle))
    y_candidate = parent_circle_seed.position.y() - int(moving_distance * sin(current_angle))
    current_pos = QPoint(x_candidate, y_candidate)

    # 计算出候选种子的像素集
    seed_pixels = get_pixels_from(image, current_pos, parent_circle_seed.radius)
    # 创建候选种子
    candidate_seed = CircleSeed(current_pos, parent_circle_seed.radius, seed_pixels, current_angle)
    candidate_seed.general_strategy = "single direction detect strategy"

    # 计算候选种子的外围条件
    peripheral_condition = calculate_peripheral_condition(candidate_seed, parent_circle_seed, detection_param)
    candidate_seed.peripheral_condition = peripheral_condition

    # 分析候选种子的外围条件
    # 条件二：相似灰度像素百分比 >= 60%
    similarity_gray_proportion = peripheral_condition.PSGP >= detection_param.SSGP
    # is_best_one = peripheral_condition.PSGP > similarity_gray_pixels_proportion

    if similarity_gray_proportion:
        candidate_seeds = candidate_seed
        # similarity_gray_pixels_proportion = peripheral_condition.PRP

    return candidate_seeds
