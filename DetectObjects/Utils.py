# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:42
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : Utils.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pow
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QColor, QImage
from .Pixels import SeedPixel
from Core.Exception import RadiusOverBorder
from .CircleSeed import CircleSeed


class GrayPixelsSet:
    """灰色像素集"""
    # def __init__(self, gray_difference: int = 40, gray_min: int = 20, gray_max=200):
    #     self._gray_difference = gray_difference if 0 < gray_difference < 200 else 40
    #     self._gray_min = gray_min if 0 < gray_min < 100 and gray_min < gray_max else 20
    #     self._gray_max = gray_max if 50 < gray_max < 255 else 200

    gray_difference = 40
    standard_gay_min = 20
    standard_gray_max = 200

    @staticmethod
    def standard_gray(gray_min=20, gray_max=200) -> QColor:
        s_gray = int((gray_min + gray_max) / 2)
        return QColor(s_gray, s_gray, s_gray)

    @staticmethod
    def is_gray(color: QColor) -> bool:
        r = color.red()
        g = color.green()
        b = color.blue()
        return GrayPixelsSet.is_gray_of_rgb(r, g, b)

    @staticmethod
    def is_gray_of_rgb(r: int, g: int, b: int):
        res1 = GrayPixelsSet.standard_gay_min < r, g, b < GrayPixelsSet.standard_gray_max
        res2 = abs(r - g), abs(r - b), abs(g - b) <= GrayPixelsSet.gray_difference
        return all(res1) and all(res2)

    @staticmethod
    def get_gray_color_from(src_colors: [QColor]) -> [QColor]:
        return [color for color in src_colors if GrayPixelsSet.is_gray(color)]

    @staticmethod
    def get_gray_pixels_from(src_pixels: [SeedPixel]) -> [SeedPixel]:
        return [pixel for pixel in src_pixels if GrayPixelsSet.is_gray(pixel.color)]


def get_pixels_from(image: QImage, position: QPoint, radius):
    """
    :param image: 源图片，该函数会根据所给的源图片、位置和半径获取这个范围内的像素集合
    :param position: 给定的圆的位置
    :param radius: 给定的半径
    :return:
    """

    if radius > min(image.width(), image.height()):
        raise RadiusOverBorder()

    result = list()
    for y in range(position.y() - radius, position.y() + radius+ 1):
        for x in range(position.x() - radius, position.y() + radius + 1):
            if pow(x - position.x(), 2) + pow(y - position.y(), 2) <= pow(radius, 2):
                pixel_color = image.pixelColor(x, y)
                result.append(SeedPixel(QPoint(x, y), pixel_color))

    return result


def calculate_reference_color_from(circle_seed: CircleSeed) -> QColor:

    seed_pixels = circle_seed.seed_pixels
    if not seed_pixels:
        return GrayPixelsSet.standard_gray()

    sum_r, sum_g, sum_b = 0, 0, 0
    for seed_pixel in seed_pixels:
        sum_r += seed_pixel.r
        sum_g += seed_pixel.g
        sum_b += seed_pixel.b

    m = len(seed_pixels)
    return QColor(int(sum_r / m), int(sum_g / m), int(sum_b / m))


if __name__ == '__main__':
    """"""
