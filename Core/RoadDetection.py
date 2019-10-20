# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 17:12
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : RoadDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QPoint
from Core.Exception import NoInitializeError
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import get_pixels_from
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters
from Core.DetectionStrategy.RoadDetectionContext import RoadDetectionContext
from Core.DetectionStrategy.GNSDetectionStrategy import GNSDetectionStrategy
from Core.DetectionStrategy.GRSDetectionStrategy import GRSDetectionStrategy
from Core.DetectionStrategy.NSDetectionStrategy import NSDetectionStrategy
from Core.DetectionStrategy.JSDetectionStrategy import JSDetectionStrategy
from Core.DetectionStrategy.SDDetectionStrategy import SDDetectionStrategy


class RoadDetection(QObject):

    circle_seeds_generated = pyqtSignal(CircleSeed)

    def __init__(self, image: QImage):
        super(RoadDetection, self).__init__()
        self._image = image
        self._seed_list = []
        self._next_circle_seed = -1
        self._road_pixels = []
        self._middle_distance = 12
        # self._angle_interval = [1, pi, 12]
        self._angle_interval = pi / 12

    def get_seed_list(self) -> list:
        return list(self._seed_list)

    def initialize(self, position: QPoint, radius: int, direction: float = 0.):
        """
        :param position:
        :param radius:
        :param direction
        :return:
        """
        circle_seed_pixels = get_pixels_from(self._image, position, radius)
        circle_seed = CircleSeed(position, radius, circle_seed_pixels, direction=direction)
        self._seed_list.append(circle_seed)
        self._next_circle_seed += 1
        print("初始种子\n", circle_seed)
        return circle_seed

    def road_detection(self):
        if len(self._seed_list) <= 0:
            raise NoInitializeError()

        road_detection_context = RoadDetectionContext()
        while not self._next_circle_seed == len(self._seed_list):
            current_circle_seed = self._seed_list[self._next_circle_seed]  # type: CircleSeed

            self._next_circle_seed += 1

            # 策略选择 单向/多向/指定多向 -- 默认多向
            print("多向道路检测...")
            detection_param = DetectionParameters.generate_multi_directional_general_similarity_detection_parameters(
                current_circle_seed.radius)
            road_detection_context.road_detection_strategy = GNSDetectionStrategy()
            general_seeds = road_detection_context.road_detect(self._image, current_circle_seed, detection_param,
                                                               self._angle_interval)
            if general_seeds:
                for general_seed in general_seeds:
                    self.add_circle_seed(general_seed)
                continue

            # # 多向策略不行，进行窄相似算法检测
            # print("窄道路检测...")
            # detection_param = DetectionParameters.generate_multi_directional_narrow_similarity_detection_parameters(
            #     current_circle_seed.radius)
            # road_detection_context.road_detection_strategy = NSDetectionStrategy()
            # general_seed = road_detection_context.road_detect(self._image, current_circle_seed, detection_param,
            #                                                   self._angle_interval)
            # if general_seed:
            #     self._seed_list.append(general_seed)
            #     self.circle_seeds_generated.emit(current_circle_seed, [general_seed])
            #     continue
            #
            # # 窄相似算法不行，进行灰度相似算法检测
            # print("灰度道路检测...")
            # detection_param = DetectionParameters.generate_multi_directional_gray_similarity_detection_parameters(
            #     current_circle_seed.radius)
            # road_detection_context.road_detection_strategy = GRSDetectionStrategy()
            # general_seed = road_detection_context.road_detect(self._image, current_circle_seed, detection_param,
            #                                                   self._angle_interval)
            # if general_seed:
            #     self._seed_list.append(general_seed)
            #     self.circle_seeds_generated.emit(current_circle_seed, [general_seed])
            #     continue

            # 灰度相似算法不行， 进行跳跃相似算法测试
            # print("跳跃道路检测...")
            # detection_param = DetectionParameters.generate_multi_directional_jump_similarity_detection_parameters(
            #     current_circle_seed.radius)
            # road_detection_context.road_detection_strategy = JSDetectionStrategy()
            # general_seed = road_detection_context.road_detect(self._image, current_circle_seed, detection_param,
            #                                                   self._angle_interval)
            # if general_seed:
            #     self._seed_list.append(general_seed)
            #     self.circle_seeds_generated.emit(current_circle_seed, [general_seed])
            #     continue

            # # 多向策略不行，进行单向检测
            # detection_param = DetectionParameters.generate_single_directional_similarity_detection_parameters(
            #     current_circle_seed.radius)
            # general_seed = single_direction_detection(self._image, current_circle_seed, detection_param,
            #                                           self._angle_interval)
            # if general_seed:
            #     self._seed_list.append(general_seed)
            #     continue

    def add_circle_seed(self, circle_seed: CircleSeed):
        if self._validation_road_pixels_proportion(circle_seed):
            self._seed_list.append(circle_seed)
            self.circle_seeds_generated.emit(circle_seed)

    def _validation_road_pixels_proportion(self, circle_seed: CircleSeed, standard_proportion: float = 0.1):
        """
        验证种子重合比例以避免产生过多相似的种子，产生死循环 ！！！！！！可以且需要优化
        :param circle_seed: 待测的候选种子
        :param standard_proportion: 待测种子与已测道路重合像素百分比阈值
        :return: bool 重合比例是否小于阈值
        """
        for road_seed in reversed(self._seed_list):
            intersected_pixels_num = circle_seed.intersected_pixel_num_with(road_seed)
            road_intersected_proportion = intersected_pixels_num / len(circle_seed.seed_pixels)
            if road_intersected_proportion > standard_proportion:
                return False
        return True

    def calculate_peripheral_condition_of(self, circle_seed):
        """TODO"""

    def _validation_adjacent_seeds(self):
        """TODO"""
