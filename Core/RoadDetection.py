# -*- coding: utf-8 -*-
# @Time    : 2019/10/4 17:12
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : RoadDetection.py
# @Project : RoadExtraction
# @Software: PyCharm

import time
from math import pi, sin, cos
import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt, QPoint, QPointF
from PyQt5.QtGui import QImage, QPainterPath, QPainter
from Core.Exception import NoInitializeError
from DetectObjects.CircleSeed import CircleSeed, CircleSeedNp
from DetectObjects.Utils import get_pixels_from, adjust_angle, get_circle_path, qimage2cvmat
from Core.DetectionStrategy.Strategy import DetectionStrategy
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters
from Core.DetectionStrategy.RoadDetectionContext import RoadDetectionContext
from Core.DetectionStrategy.GNSDetectionStrategy import GNSDetectionStrategy
from Core.DetectionStrategy.SDDetectionStrategy import SDDetectionStrategy
from Core.DetectionStrategy.JSDetectionStrategy import JSDetectionStrategy


class RoadDetection(QObject):
    """
    道路提取。进行道路跟踪提取的主要类，封装了对高分辨率遥感图像
    进行道路半自动道路跟踪提取的所有功能：道路跟踪、形态学过滤、
    道路中心线划分等
    """
    circle_seeds_generated = pyqtSignal(CircleSeedNp)

    def __init__(self, image: np.ndarray):
        super(RoadDetection, self).__init__()
        self._image = image
        self._seed_list = []
        self._next_circle_seed = -1
        self._road_img = np.zeros(self._image.shape[:2], np.int8)

        self._angle_interval = pi / 12
        self._is_init = False
        self._is_generated_circle_seed = False

        self._road_path = QPainterPath
        self._run_time_data = {
            "SSD": [],
            "GNSD": [],
            "JSD": [],
            "getPos": [],
            "validate": []
        }

    def get_run_time_data(self) -> dict:
        return self._run_time_data

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, new_image: np.ndarray):
        if isinstance(new_image, np.ndarray):
            self._image = new_image
        else:
            raise TypeError("图片使用numpy的ndarray")

    def get_seed_list(self) -> list:
        return list(self._seed_list)

    def set_init(self, init):
        self._is_init = init

    def initialize(self, position: QPoint, radius: int, direction: float = 0.):
        """
        初始化第一个圆形种子（人工确定）
        :param position: 种子的位置
        :param radius: 种子的半径
        :param direction 种子的方向
        """
        if isinstance(position, QPointF):
            position = position.toPoint()
        position = [position.x(), position.y()]

        circle_seed = CircleSeedNp(position, radius, DetectionStrategy.Initialization, self._image, direction)
        self._seed_list.append(circle_seed)
        self._next_circle_seed += 1
        self._is_init = True
        return circle_seed

    def initialize_by_circle_seed(self, circle_seed: CircleSeedNp):
        self._seed_list.append(circle_seed)
        self._next_circle_seed += 1
        self._is_init = True
        return circle_seed

    def road_detection_one_step(self):
        """遥感图像道路跟踪的一步（用来测试）"""
        if not self._is_init:
            raise NoInitializeError()

        road_detection_context = RoadDetectionContext()
        if not self._next_circle_seed == len(self._seed_list):
            current_circle_seed = self._seed_list[self._next_circle_seed]  # type: CircleSeed
            self._next_circle_seed += 1

            st = time.time()
            # 单向和多向
            single_direction_detect_param = DetectionParameters.get_detection_parameters(
                DetectionStrategy.SingleSSDetectionStrategy, current_circle_seed.radius)
            road_detection_context.road_detection_strategy = SDDetectionStrategy()
            single_analysis_seeds = road_detection_context.road_detection(
                self._image, current_circle_seed, self._seed_list[0],
                self._angle_interval,
                DetectionStrategy.SingleSSDetectionStrategy,
                single_direction_detect_param)
            self._run_time_data["SSD"].append(time.time() - st)

            st = time.time()
            multi_direction_detect_param = DetectionParameters.get_detection_parameters(
                DetectionStrategy.MultiGNSDetectionStrategy, current_circle_seed.radius)
            road_detection_context.road_detection_strategy = GNSDetectionStrategy()
            multi_analysis_seeds = road_detection_context.road_detection(
                self._image, current_circle_seed, self._seed_list[0],
                self._angle_interval,
                DetectionStrategy.MultiGNSDetectionStrategy,
                multi_direction_detect_param)
            self._run_time_data["GNSD"].append(time.time() - st)

            # 对单向和多向检测结果进行决策
            if single_analysis_seeds or multi_analysis_seeds:
                if single_analysis_seeds and multi_analysis_seeds:
                    single_analysis_seed = single_analysis_seeds[0]
                    self.add_circle_seed(single_analysis_seed)
                    for candidate_seed in multi_analysis_seeds:  # type: CircleSeedNp
                        if abs(candidate_seed.direction - single_analysis_seed.direction) >= pi / 6:
                            self.add_circle_seed(candidate_seed)
                elif single_analysis_seeds:
                    self.add_circle_seed(single_analysis_seeds[0])
                else:
                    for candidate_seed in multi_analysis_seeds:  # type: CircleSeedNp
                        self.add_circle_seed(candidate_seed)
            # 跳跃性检测
            if not self._is_generated_circle_seed:
                # print(self._next_circle_seed - 1)
                st = time.time()
                multi_jump_detect_param = DetectionParameters.get_detection_parameters(
                    DetectionStrategy.MultiJSDetectionStrategy, current_circle_seed.radius)
                road_detection_context.road_detection_strategy = JSDetectionStrategy()
                multi_jump_detect_param.MD += multi_jump_detect_param.DI
                # print(multi_jump_detect_param.MD, ", ", multi_jump_detect_param.MMD)
                while not (self._is_generated_circle_seed or multi_jump_detect_param.MD >= multi_jump_detect_param.MMD):
                    # print(multi_jump_detect_param.MD, ", ", multi_jump_detect_param.MMD)
                    jump_detected_seeds = road_detection_context.road_detection(
                        self._image, current_circle_seed, self._seed_list[0],
                        self._angle_interval,
                        DetectionStrategy.MultiJSDetectionStrategy,
                        multi_jump_detect_param)
                    if jump_detected_seeds:
                        self.add_circle_seed(jump_detected_seeds[0])
                    multi_jump_detect_param.MD += multi_jump_detect_param.DI
                self._run_time_data["JSD"].append(time.time() - st)
            self._is_generated_circle_seed = False
            return True
        else:
            self._is_init = False
            return False

    def _generate_candidate_seeds(self, parent_seed, detection_strategy, detection_param):
        """
        根据父种子、检测策略和有效角度范围生成候选种子（未经筛选）。
        当不指定有效有效角度范围时，会沿着父种子的方向生成一个候选种子
        :param parent_seed: 用于生成候选种子的父种子
        :type parent_seed: CircleSeed
        :param detection_strategy: 道路跟踪检测的策略
        :type detection_strategy: DetectionStrategy
        :param detection_param: 道路跟踪检测的常数参数
        :type detection_param: DetectionParameters
        :return: 返回生成的候选种子列表
        :rtype: list
        """

        def create_candidate_seed(current_angle: float) -> [CircleSeedNp, None]:
            x_candidate = parent_seed.position[0] + int(moving_distance * cos(current_angle))
            y_candidate = parent_seed.position[1] - int(moving_distance * sin(current_angle))

            bound_x = (x_candidate - parent_seed.radius < 0, x_candidate + parent_seed.radius > self._image.shape[1])
            bound_y = (y_candidate - parent_seed.radius < 0, y_candidate + parent_seed.radius > self._image.shape[0])
            if any(bound_x) or any(bound_y):
                return None
            current_pos = [x_candidate, y_candidate]

            # 创建候选种子
            candidate_seed = CircleSeedNp(current_pos, parent_seed.radius, detection_strategy,
                                          self._image, current_angle, parent_seed=self._seed_list[0])
            return candidate_seed
        if detection_param.ERA is None:
            return [create_candidate_seed(parent_seed.direction)]
        moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_seed.radius
        candidate_seeds = []
        half_era = detection_param.ERA / 2

        for k in range(0, int(half_era / self._angle_interval) + 1):
            pos_current_angle = adjust_angle(k * self._angle_interval + parent_seed.direction)
            nes_current_angle = adjust_angle(parent_seed.direction - k * self._angle_interval)

            circle_seed = create_candidate_seed(pos_current_angle)
            if circle_seed:
                candidate_seeds.append(circle_seed)
            if k != 0:
                circle_seed = create_candidate_seed(nes_current_angle)
                if circle_seed:
                    candidate_seeds.append(circle_seed)
        return candidate_seeds

    def road_detection(self):
        if len(self._seed_list) <= 0:
            raise NoInitializeError()

        while not self._next_circle_seed == len(self._seed_list):
            current_circle_seed = self._seed_list[self._next_circle_seed]  # type: CircleSeed
            self._next_circle_seed += 1

            # 先进行单向检测 再进行多方向检测
            if not self.single_direction_detect_approach(current_circle_seed):
                self.multi_directional_detect_approach(current_circle_seed)

        # 检测完毕， 进行形态学处理：进行形态学开操作和闭操作
        self.morphologically_filtering()

        # 至此道路基本被提取出来了，最后进行道路中心线划分
        self.road_centre_lines()

        # 最后进行道路建模，并返回道路模型（以便对道路进行各种格式进行转换导出）
        return self.build_road_model()

    def add_circle_seed(self, circle_seed: CircleSeedNp, standard_proportion=0.1):
        """
           验证种子重合比例以避免产生过多相似的种子，产生死循环 ！！！！！！可以且需要优化
           :param circle_seed: 待测的候选种子
           :param standard_proportion: 待测种子与已测道路重合像素百分比阈值
           :return: bool 重合比例是否小于阈值
        """
        result = True
        st = time.time()
        circle_pixels = circle_seed.get_pixels_position(self._image.shape)
        self._run_time_data["getPos"].append(time.time() - st)

        if standard_proportion is not None and standard_proportion > 0:
            st = time.time()
            road_pixels = self._road_img[circle_pixels[:, 1], circle_pixels[:, 0]]
            intersect_num = road_pixels[road_pixels == 1].size
            road_pixels_proportion = intersect_num / circle_pixels.shape[0]
            result = road_pixels_proportion <= standard_proportion
            dt = time.time() - st
            self._run_time_data["validate"].append(dt)
            print("validate ", dt)

        if result:
            self._is_generated_circle_seed = True
            self._seed_list.append(circle_seed)
            self._road_img[circle_pixels[:, 1], circle_pixels[:, 0]] = 1
            self._seed_list[self._next_circle_seed - 1].append_child_seed(circle_seed)
            self.circle_seeds_generated.emit(circle_seed)

    def validation_road_pixels_proportion(self, circle_seed: CircleSeedNp, standard_proportion: float = 0.12):
        """
        验证种子重合比例以避免产生过多相似的种子，产生死循环 ！！！！！！可以且需要优化
        :param circle_seed: 待测的候选种子
        :param standard_proportion: 待测种子与已测道路重合像素百分比阈值
        :return: bool 重合比例是否小于阈值
        """
        # circle_pixels = circle_seed.get_pixels_position(self._image.shape)
        # intersect_pos = np.intersect1d(self._road_pixels, circle_pixels)
        # intersect_num = intersect_pos.size
        # road_pixels_proportion = intersect_num / circle_pixels.size
        # result = road_pixels_proportion <= standard_proportion
        #
        # res = []
        # for index, seed in enumerate(self._seed_list):  # type: CircleSeedNp
        #     pos = seed.get_pixels_position(self._image.shape)
        #     intersect_pos1 = np.intersect1d(circle_pixels, pos)
        #     if intersect_pos1.size > 0:
        #         print(index, "号种子： ", intersect_pos1.size)
        #         print("交叉坐标: \n", intersect_pos1)
        #         print("种子坐标：\n", )
        #         res.append(index)
        # print(intersect_pos)
        # print("全局交叉道路坐标： ", circle_pixels.size, ", ", intersect_pos.size)
        # print("道路像素比例： ", road_pixels_proportion)
        # return res

    def road_extraction(self):
        """
        道路跟踪检测后进行道路的提取：
        图像二值化--道路部分为白色，非道路部分为黑色
        形态学过滤--进行开操作把圆形种子之间的缝隙填上
        ....(更多优化的方案)
        道路中心线的划分
        道路模型的建立
        导出结果 ---- 矢量图，普通的像素图、数据表格
        """
        # 图像二值化
        # 创建一个空白的黑色图片
        road_image = QImage(self._image.shape[1], self._image.shape[0], QImage.Format_RGB888)
        painter = QPainter(road_image)

        for circle_seed in self._seed_list:  # type: CircleSeedNp
            path = get_circle_path(circle_seed.position, circle_seed.radius)
            painter.fillPath(path, Qt.white)

        cv_image = qimage2cvmat(road_image)
        cv2.imshow("road image", cv_image)

        # 形态学过滤
        morph_filtered_img = self.morphologically_filtering(cv_image)
        cv2.imshow("morph_filtered road image", morph_filtered_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # 道路中心线的划分
        self.road_centre_lines()

        # 建立道路模型
        self.build_road_model()

    def morphologically_filtering(self, cv_image: np.ndarray):
        morph_filtered_img = cv2.morphologyEx(cv_image, cv2.MORPH_OPEN, kernel=(5, 5), iterations=8)
        return morph_filtered_img

    def road_centre_lines(self):
        """TODO"""

    def build_road_model(self):
        """TODO"""

    def _validation_adjacent_seeds(self):
        """TODO"""


class RoadDetectionEx(QObject):
    circle_seeds_generated = pyqtSignal(CircleSeed)

    def __init__(self, image: QImage = None):
        super(RoadDetectionEx, self).__init__()
        self._image = image
        self._seed_list = []
        self._road_pixels = []
        self._next_circle_seed = 0
        self._middle_distance = 12
        self._angle_interval = pi / 12
        self._is_init = False

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, new_image):
        if isinstance(new_image, QImage):
            self._image = new_image
        else:
            raise TypeError("无效类型：{}, 请使用类型：{}".format(type(new_image), QImage.__name__))

    def get_seed_list(self) -> list:
        return list(self._seed_list)

    def initialize(self, position: QPoint, radius: int, direction: float = 0.):
        """
        初始化开始的种子
        :param position: 位置
        :type position: QPoint
        :param radius: 半径（像素）
        :type radius: int
        :param direction 方向（0-2π）
        :type direction: float
        """
        circle_seed_pixels = get_pixels_from(self._image, position, radius)
        circle_seed = CircleSeed(position, radius, circle_seed_pixels, direction=direction)
        self._seed_list.append(circle_seed)
        self._road_pixels.extend(circle_seed_pixels)
        self._is_init = True
        return circle_seed

    def road_detection_one_step(self):
        if not self._is_init:
            raise NoInitializeError()

        road_detection_context = RoadDetectionContext()
        if not self._next_circle_seed == len(self._seed_list):
            current_circle_seed = self._seed_list[self._next_circle_seed]  # type: CircleSeed
            self._next_circle_seed += 1

            # 单向和多向
            single_direction_detect_param = DetectionParameters.get_detection_parameters(
                DetectionStrategy.SingleSSDetectionStrategy, current_circle_seed.radius)
            single_candidate_seeds = self._generate_candidate_seeds(
                current_circle_seed, DetectionStrategy.SingleSSDetectionStrategy, single_direction_detect_param)
            road_detection_context.road_detection_strategy = SDDetectionStrategy()
            single_analysis_seeds = road_detection_context.analysis_peripheral_condition(
                single_candidate_seeds, single_direction_detect_param)

            multi_direction_detect_param = DetectionParameters.get_detection_parameters(
                DetectionStrategy.MultiGNSDetectionStrategy, current_circle_seed.radius)
            multi_candidate_seeds = self._generate_candidate_seeds(
                current_circle_seed, DetectionStrategy.MultiGNSDetectionStrategy, multi_direction_detect_param)
            road_detection_context.road_detection_strategy = GNSDetectionStrategy()
            multi_analysis_seeds = road_detection_context.analysis_peripheral_condition(
                multi_candidate_seeds, multi_direction_detect_param)

            # 对单向和多向检测结果进行决策
            if single_analysis_seeds or multi_analysis_seeds:
                if single_analysis_seeds and multi_analysis_seeds:
                    single_analysis_seed = single_analysis_seeds[0]
                    self.add_circle_seed(single_analysis_seed)
                    for candidate_seed in multi_analysis_seeds:  # type: CircleSeed
                        if abs(candidate_seed.direction - single_analysis_seed.direction) >= pi / 6:
                            self.add_circle_seed(candidate_seed, validation_rpp=True)
                elif single_analysis_seeds:
                    self.add_circle_seed(single_analysis_seeds[0])
                else:
                    for candidate_seed in multi_analysis_seeds:  # type: CircleSeed
                        self.add_circle_seed(candidate_seed, validation_rpp=True)
                return True
            return True
        else:
            self._is_init = False
            return False

    def _generate_candidate_seeds(self, parent_seed, detection_strategy, detection_param, effective_range_angle=True):
        """
        根据父种子、检测策略和有效角度范围生成候选种子（未经筛选），有效角度范围默认为None。
        当不指定有效有效角度范围时，会沿着父种子的方向生成一个候选种子
        :param parent_seed: 用于生成候选种子的父种子
        :type parent_seed: CircleSeed
        :param detection_strategy: 道路跟踪检测的策略
        :type detection_strategy: DetectionStrategy
        :param detection_param: 道路跟踪检测的常数参数
        :type detection_param: DetectionParameters
        :param effective_range_angle:  ERA有效角度范围，限制候选种子的方向
        :type effective_range_angle: float
        :return: 返回生成的候选种子列表
        :rtype: list
        """

        def create_candidate_seed(current_angle: float) -> CircleSeed:
            x_candidate = parent_seed.position.x() + int(moving_distance * cos(current_angle))
            y_candidate = parent_seed.position.y() - int(moving_distance * sin(current_angle))
            current_pos = QPoint(x_candidate, y_candidate)

            # 计算出候选种子的像素集
            seed_pixels = get_pixels_from(self._image, current_pos, parent_seed.radius)
            # 创建候选种子
            candidate_seed = CircleSeed(current_pos, parent_seed.radius, seed_pixels, current_angle, parent_seed=self._seed_list[0])
            candidate_seed.general_strategy = detection_strategy
            return candidate_seed

        if not effective_range_angle:
            return [create_candidate_seed(parent_seed.direction)]

        moving_distance = detection_param.MD if detection_param.MD > 0 else 2 * parent_seed.radius
        candidate_seeds = []
        half_era = detection_param.ERA / 2

        for k in range(0, int(half_era / self._angle_interval)):
            pos_current_angle = adjust_angle(k * self._angle_interval + parent_seed.direction)
            nes_current_angle = adjust_angle(parent_seed.direction - k * self._angle_interval)
            candidate_seeds.append(create_candidate_seed(pos_current_angle))
            if k != 0:
                candidate_seeds.append(create_candidate_seed(nes_current_angle))
        return candidate_seeds

    def road_detection(self):
        if len(self._seed_list) <= 0:
            raise NoInitializeError()

        while not self._next_circle_seed == len(self._seed_list):
            current_circle_seed = self._seed_list[self._next_circle_seed]  # type: CircleSeed
            self._next_circle_seed += 1

            # 先进行单向检测 再进行多方向检测
            if not self.single_direction_detect_approach(current_circle_seed):
                self.multi_directional_detect_approach(current_circle_seed)

        # 检测完毕， 进行形态学处理：进行形态学开操作和闭操作
        self.morphologically_filtering()

        # 至此道路基本被提取出来了，最后进行道路中心线划分
        self.road_centre_lines()

        # 最后进行道路建模，并返回道路模型（以便对道路进行各种格式进行转换导出）
        return self.build_road_model()

    def morphologically_filtering(self):
        """TODO"""

    def road_centre_lines(self):
        """TODO"""

    def build_road_model(self):
        """TODO"""

    def add_circle_seed(self, circle_seed: CircleSeed, validation_rpp=False):
        if not validation_rpp or self._validation_road_pixels_proportion(circle_seed, 0.1):
            self._seed_list.append(circle_seed)
            self._road_pixels.extend(circle_seed.seed_pixels)
            self._seed_list[self._next_circle_seed - 1].append_child_seed(circle_seed)
            self.circle_seeds_generated.emit(circle_seed)

    def _validation_road_pixels_proportion(self, circle_seed: CircleSeed, standard_proportion: float = 0.1):
        """
        验证种子重合比例以避免产生过多相似的种子，产生死循环 ！！！！！！可以且需要优化
        :param circle_seed: 待测的候选种子
        :param standard_proportion: 待测种子与已测道路重合像素百分比阈值
        :return: bool 重合比例是否小于阈值
        """
        t = time.time()
        intersected_pixels_num = len(list(set(circle_seed.seed_pixels) & set(self._road_pixels)))
        road_intersected_proportion = intersected_pixels_num / len(circle_seed.seed_pixels)
        print("交集运算时间： ", time.time() - t)
        if road_intersected_proportion > standard_proportion:
            return False
        return True

    def _validation_adjacent_seeds(self):
        """TODO"""
