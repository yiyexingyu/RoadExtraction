# -*- coding: utf-8 -*-
# @Time    : 2019/10/19 18:27
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : GNSDetectionStrategy.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5.QtGui import QImage
from DetectObjects.Utils import calculate_spectral_distance
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters
from Core.DetectionAlgorithm.GeneralSimilarityDetection import general_similarity_detection
from DetectObjects.CircleSeed import CircleSeed, CircleSeedNp
from .AbstractDetectionStrategy import AbstractDetectionStrategy


class GNSDetectionStrategy(AbstractDetectionStrategy):

    def __init__(self):
        super(GNSDetectionStrategy, self).__init__()

    def road_detect(self, image: QImage, parent_seed: CircleSeed, detection_param: DetectionParameters, angle_interval):
        return general_similarity_detection(image, parent_seed, detection_param, angle_interval)

    def analysis_peripheral_condition(self, candidate_seeds: list, detection_param: DetectionParameters):
        """
        候选种子的验证
        :param candidate_seeds: 待验证的候选种子
        :param detection_param: 校验常量参数
        :return: 返回通过验证的候选种子
        """

        result = []

        for candidate_seed in candidate_seeds:  # type: CircleSeedNp
            if candidate_seed.spectral_distance <= detection_param.SSD:
                result.append(candidate_seed)

        # 校验邻居候选种子(与生成的种子方向相距60°的候选种子，以检测道路是否变宽
        # 如果其邻居种子也是符合条件的种子，那继续校验邻居种子的邻居种子，直至没有
        return reversed(sorted(result))
