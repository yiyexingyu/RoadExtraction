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
from DetectObjects.CircleSeed import CircleSeed, CircleSeedNp
from .AbstractDetectionStrategy import AbstractDetectionStrategy


class SDDetectionStrategy(AbstractDetectionStrategy):

    def __init__(self):
        super(SDDetectionStrategy, self).__init__()

    def road_detect(self, image: QImage, parent_seed: CircleSeed, detection_param: DetectionParameters, angle_interval):
        return single_direction_detection(image, parent_seed, detection_param, angle_interval)

    def analysis_peripheral_condition(self, candidate_seeds: list, detection_param: DetectionParameters):
        """
        候选种子的验证
        :param candidate_seeds: 待验证的候选种子
        :param detection_param: 校验常量参数
        :return: 返回通过验证的候选种子
        """
        temp_result = None
        for index, candidate_seed in enumerate(candidate_seeds):  # type: int, CircleSeedNp
            if candidate_seed.spectral_distance <= detection_param.SSD:
                if index == 0:
                    return [candidate_seed]
                if not temp_result:
                    temp_result = candidate_seed
                elif temp_result.spectral_distance > candidate_seed.spectral_distance:
                    temp_result = candidate_seed
        return [temp_result] if temp_result else []

