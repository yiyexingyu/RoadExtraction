# -*- coding: utf-8 -*-
# @Time    : 2019/10/6 22:09
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : PeripheralCondition.py
# @Project : RoadExtraction
# @Software: PyCharm

# from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import GrayPixelsSet
from Core.DetectionAlgorithm.DetectionParameters import DetectionParameters


class PeripheralCondition:

    def __init__(self,
                 similarity_gray_pixels_proportion: float,
                 gray_pixels_proportion: float,
                 addition_gray_proportion: float,
                 road_pixels_proportion: float,
                 direction: float):
        self._similarity_gray_pixels_proportion = similarity_gray_pixels_proportion
        self._gray_pixels_proportion = gray_pixels_proportion
        self._addition_gray_proportion = addition_gray_proportion
        self._road_pixels_proportion = road_pixels_proportion
        self._direction = direction

    @property
    def PSGP(self):
        return self._similarity_gray_pixels_proportion

    @property
    def PGP(self):
        return self._gray_pixels_proportion

    @property
    def AGP(self):
        return self._addition_gray_proportion

    @property
    def PRP(self):
        return self._road_pixels_proportion

    @property
    def direction(self):
        return self._direction

    def __str__(self):
        psgp = "PSPG = " + str(self.PSGP) + "\n"
        pgp = "PGP = " + str(self.PGP) + "\n"
        agp = "AGP = " + str(self.AGP) + "\n"
        prp = "PRP = " + str(self.PRP) + "\n"
        return "外围条件： \n" + psgp + pgp + agp + prp


def calculate_peripheral_condition(
        circle_seed, parent_seed, detection_param: DetectionParameters) -> PeripheralCondition:
    similarity_gray_pixels = GrayPixelsSet.get_similarity_gray_pixels(
        circle_seed.seed_pixels, parent_seed.reference_color, detection_param.CDiff)
    similarity_gray_pixels_proportion = len(similarity_gray_pixels) / len(circle_seed.seed_pixels)
    gray_pixels_proportion = len(circle_seed.gray_pixels) / len(circle_seed.seed_pixels)

    # PRP ??????
    road_pixels_proportion = 0.
    addition_gray_proportion = similarity_gray_pixels_proportion - road_pixels_proportion

    peripheral_condition = PeripheralCondition(similarity_gray_pixels_proportion, gray_pixels_proportion,
                                               addition_gray_proportion, road_pixels_proportion, parent_seed.direction)
    return peripheral_condition
