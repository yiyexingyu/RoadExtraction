# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 14:50
# @Author  : 一叶星羽
# @Email   : 2958029539@qq.com
# @File    : DetectionParameters.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi
from Core.DetectionStrategy.Strategy import DetectionStrategy
NO_LIMIT = float("inf")


class DetectionParameters:

    def __init__(self,
                 standard_similar_gray_proportion: [float, None],
                 addition_gray_proportion: [float, None],
                 standard_gray_proportion: [float, None],
                 standard_road_proportion: [float, None],
                 color_difference: [int, None],
                 effective_range_angle: float,
                 is_detect_neighbour: bool,
                 moving_distance: int,
                 max_moving_distance: [int, None],
                 max_number_generated_seeds: [int, float],
                 distance_increment: [int, None],
                 spectral_distance: float,
                 texture_distance: float):
        self._standard_similar_gray_proportion = standard_similar_gray_proportion
        self._addition_gray_proportion = addition_gray_proportion
        self._standard_gray_proportion = standard_gray_proportion
        self._standard_road_proportion = standard_road_proportion
        self._color_difference = color_difference
        self._effective_range_angle = effective_range_angle
        self._is_detect_neighbour = is_detect_neighbour
        self._moving_distance = moving_distance
        self._max_moving_distance = max_moving_distance
        self._max_number_generated_seeds = max_number_generated_seeds
        self._distance_increment = distance_increment

        self._standard_spectral_distance = spectral_distance
        self._standard_texture_distance = texture_distance

    @property
    def SSGP(self):
        return self._standard_similar_gray_proportion

    @property
    def AGP(self):
        return self._addition_gray_proportion

    @property
    def SGP(self):
        return self._standard_gray_proportion

    @property
    def SRP(self):
        return self._standard_road_proportion

    @property
    def CDiff(self):
        return self._color_difference

    @CDiff.setter
    def CDiff(self, color_diff):
        self._color_difference = color_diff

    @property
    def ERA(self):
        return self._effective_range_angle

    @ERA.setter
    def ERA(self, era):
        self._effective_range_angle = era

    @property
    def MD(self):
        return self._moving_distance

    @MD.setter
    def MD(self, new_md):
        self._moving_distance = new_md

    @property
    def MMD(self):
        return self._max_moving_distance

    @property
    def DI(self):
        """距离增量"""
        return self._distance_increment

    @property
    def is_detect_neighbour(self) -> bool:
        return self._is_detect_neighbour

    @property
    def max_number_generated_seeds(self):
        return self._max_number_generated_seeds

    @property
    def SSD(self):
        """标准光谱特征距离"""
        return self._standard_spectral_distance

    @SSD.setter
    def SSD(self, spectral_distance):
        self._standard_spectral_distance = spectral_distance

    @property
    def STD(self):
        """标准纹理特征距离"""
        return self._standard_texture_distance

    @STD.setter
    def STD(self, texture_distance):
        self._standard_texture_distance = texture_distance

    @staticmethod
    def get_detection_parameters(detection_strategy: DetectionStrategy, radius: int):
        if detection_strategy == DetectionStrategy.MultiGNSDetectionStrategy:
            return DetectionParameters.generate_multi_directional_general_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.MultiGRSDetectionStrategy:
            return DetectionParameters.generate_multi_directional_gray_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.MultiNSDetectionStrategy:
            return DetectionParameters.generate_multi_directional_narrow_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.MultiJSDetectionStrategy:
            return DetectionParameters.generate_multi_directional_jump_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.SingleSSDetectionStrategy:
            return DetectionParameters.generate_single_directional_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.SingleGRSDetectionStrategy:
            return DetectionParameters.generate_single_directional_gray_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.SingleNSDetectionStrategy:
            return DetectionParameters.generate_single_directional_narrow_similarity_detection_parameters(radius)
        elif detection_strategy == DetectionStrategy.SingleJSDetectionStrategy:
            return DetectionParameters.generate_single_directional_jump_similarity_detection_parameters(radius)
        else:
            raise NotImplemented

    @staticmethod
    def generate_multi_directional_general_similarity_detection_parameters(radius: int):
        """多方向检测的一般相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=0.7,  # 0.7
            addition_gray_proportion=0.7,
            standard_gray_proportion=None,
            standard_road_proportion=None,
            color_difference=44,    # 25,
            effective_range_angle=11 * pi / 6,
            is_detect_neighbour=True,
            max_number_generated_seeds=NO_LIMIT,
            moving_distance=2 * radius,
            max_moving_distance=None,
            distance_increment=None,

            spectral_distance=40.6,   #65
            texture_distance=3.6
        )

    @staticmethod
    def generate_multi_directional_jump_similarity_detection_parameters(radius: int):
        """多方向检测的跳跃相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=0.25,
            addition_gray_proportion=None,
            standard_gray_proportion=None,
            standard_road_proportion=0.1,
            color_difference=15,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=8 * radius,
            distance_increment=6,

            spectral_distance=50.6,  #76
            texture_distance=1.2
        )

    @staticmethod
    def generate_single_directional_similarity_detection_parameters(radius: int):
        """单方向检测的窄相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=0.6,
            addition_gray_proportion=None,
            standard_gray_proportion=None,
            standard_road_proportion=None,
            color_difference=35,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=None,
            distance_increment=None,

            spectral_distance=40.8,  #56.6,
            texture_distance=0.6
        )











    @staticmethod
    def generate_multi_directional_gray_similarity_detection_parameters(radius: int):
        """多方向检测的窄相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=None,
            addition_gray_proportion=None,
            standard_gray_proportion=0.9,
            standard_road_proportion=0.1,
            color_difference=None,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=None,
            distance_increment=None,

            spectral_distance=26.6,
            texture_distance=3.6
        )

    @staticmethod
    def generate_multi_directional_narrow_similarity_detection_parameters(radius: int):
        """多方向检测的窄相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=0.3,
            addition_gray_proportion=None,
            standard_gray_proportion=None,
            standard_road_proportion=0.1,
            color_difference=15,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=None,
            distance_increment=None,

            spectral_distance=56.6,
            texture_distance=3.45
        )

    @staticmethod
    def generate_single_directional_narrow_similarity_detection_parameters(radius: int):
        """单方向检测的窄相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=0.21,
            addition_gray_proportion=None,
            standard_gray_proportion=None,
            standard_road_proportion=0.1,
            color_difference=15,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=None,
            distance_increment=None,

            spectral_distance=76.6,
            texture_distance=0.6
        )

    @staticmethod
    def generate_single_directional_gray_similarity_detection_parameters(radius: int):
        """单方向检测的窄相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=None,
            addition_gray_proportion=None,
            standard_gray_proportion=0.6,
            standard_road_proportion=0.1,
            color_difference=None,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=None,
            distance_increment=None,

            spectral_distance=66.6,
            texture_distance=3.66
        )

    @staticmethod
    def generate_single_directional_jump_similarity_detection_parameters(radius: int):
        """单方向检测的跳跃相似性检测算法参数"""
        return DetectionParameters(
            standard_similar_gray_proportion=0.15,
            addition_gray_proportion=None,
            standard_gray_proportion=None,
            standard_road_proportion=0.1,
            color_difference=25,
            effective_range_angle=pi / 6,
            is_detect_neighbour=False,
            max_number_generated_seeds=1,
            moving_distance=2 * radius,
            max_moving_distance=6 * radius,
            distance_increment=6,

            spectral_distance=56.6,
            texture_distance=0.6
        )

