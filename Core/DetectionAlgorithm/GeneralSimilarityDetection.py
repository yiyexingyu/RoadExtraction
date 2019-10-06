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


def general_similarity_detection(image: QImage, parent_circle_seed: CircleSeed, middle_distance=-1, angle_interval=pi/12):
    """
    :param image: 源遥感图片
    :param parent_circle_seed: 父圆形种子
    :param middle_distance: 父种子和子种子中心间的距离
    :param angle_interval:  角度间距
    :return: 有效的候选种子数组
    """

    middle_distance = middle_distance if middle_distance > 0 else 2 * parent_circle_seed.radius
    candidate_seeds = []

    for k in range(1, int(2 * pi / angle_interval)):
        current_angle = k * angle_interval
        x_candidate = parent_circle_seed.position.x() + int(middle_distance * cos(current_angle))
        y_candidate = parent_circle_seed.position.y() + int(middle_distance * sin(current_angle))
        current_pos = QPoint(x_candidate, y_candidate)

        seed_pixels = get_pixels_from(image, current_pos, parent_circle_seed.radius)
        candidate_seed = CircleSeed(current_pos, parent_circle_seed.radius, seed_pixels, current_angle)

        candidate_seeds.append(candidate_seed)
