# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 16:28
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : GeneralSimilarityDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi, cos, sin
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import get_pixels_from
from .DetectionParameters import DetectionParameters
from ..PeripheralCondition import calculate_peripheral_condition


def general_similarity_detection(
        image: QImage, parent_circle_seed: CircleSeed, detection_param: DetectionParameters, angle_interval=pi / 12):
    """
    :param image: 源遥感图片
    :param parent_circle_seed: 父圆形种子
    :param detection_param: 检测算法所需参数
    :param angle_interval:  角度间距
    :return: 有效的候选种子数组
    """

    moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_circle_seed.radius
    candidate_seeds = []

    for k in range(0, int(2 * pi / angle_interval)):
        current_angle = k * angle_interval + parent_circle_seed.direction
        if current_angle > 2 * pi:
            current_angle = current_angle % (2 * pi)
        pos_relative_angle = abs(current_angle - parent_circle_seed.direction)
        neg_relative_angle = abs(2 * pi - pos_relative_angle)

        # 筛选条件6： 要求在有效角度范围内
        if min(pos_relative_angle, neg_relative_angle) > detection_param.ERA / 2:
            continue

        x_candidate = parent_circle_seed.position.x() + int(moving_distance * cos(current_angle))
        y_candidate = parent_circle_seed.position.y() + int(moving_distance * sin(current_angle))
        current_pos = QPoint(x_candidate, y_candidate)

        # 计算出候选种子的像素集
        seed_pixels = get_pixels_from(image, current_pos, parent_circle_seed.radius)
        # 创建候选种子
        candidate_seed = CircleSeed(current_pos, parent_circle_seed.radius, seed_pixels, current_angle)

        # 计算候选种子的外围条件
        peripheral_condition = calculate_peripheral_condition(candidate_seed, parent_circle_seed, detection_param)

        print("k = ", k, "  angle = ", current_angle, " parent(", + parent_circle_seed.position.x(),
              parent_circle_seed.position.y(), ")")
        print(peripheral_condition)

        # 分析候选种子的外围条件
        # 条件二：相似灰度像素百分比 >= 70%
        similarity_gray_proportion = peripheral_condition.PSGP >= detection_param.SSGP
        # 条件三： 附加灰色像素比例 >= 70%
        addition_gray_proportion = peripheral_condition.AGP >= detection_param.AGP

        if similarity_gray_proportion and addition_gray_proportion:
            candidate_seeds.append(candidate_seed)
            # print("k = ", k, "  angle = ", current_angle)

    # 条件四：相邻候选种子之间的角度差 <= AI(angle interval) 实际上只需要验证第一个和最后一个即可
    if len(candidate_seeds) >= 3:
        pos_relative_angle = abs(candidate_seeds[0].direction - candidate_seeds[-1].direction)
        neg_relative_angle = abs(2 * pi - pos_relative_angle)
        if min(pos_relative_angle, neg_relative_angle) > angle_interval:
            # 这个条件需要改善！！！！！！！！！！！！！！！！！！！！！！！！
            pop_index = 0 if len(candidate_seeds[0].gray_pixels) < len(candidate_seeds[-1].gray_pixels) else -1
            candidate_seeds.pop(pop_index)

    return candidate_seeds
