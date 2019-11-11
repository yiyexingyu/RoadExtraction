# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:42
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : Utils.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pow, sqrt, pi
import numpy as np
import cv2
from PyQt5.QtCore import QPoint, QRectF
from PyQt5.QtGui import QColor, QImage, QPainterPath
from DetectObjects.Pixels import SeedPixel
from Core.Exception import RadiusOverBorder


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
        res11 = GrayPixelsSet.standard_gay_min < r < GrayPixelsSet.standard_gray_max
        res12 = GrayPixelsSet.standard_gay_min < g < GrayPixelsSet.standard_gray_max
        res13 = GrayPixelsSet.standard_gay_min < b < GrayPixelsSet.standard_gray_max
        res1 = (res11, res12, res13)

        res21 = abs(r - g) <= GrayPixelsSet.gray_difference
        res22 = abs(r - b) <= GrayPixelsSet.gray_difference
        res23 = abs(g - b) <= GrayPixelsSet.gray_difference
        res2 = (res21, res22, res23)
        return all(res1) and all(res2)

    @staticmethod
    def get_gray_color_from(src_colors: [QColor]) -> [QColor]:
        return [color for color in src_colors if GrayPixelsSet.is_gray(color)]

    @staticmethod
    def get_gray_pixels_from(src_pixels: [SeedPixel]) -> [SeedPixel]:
        return [pixel for pixel in src_pixels if GrayPixelsSet.is_gray(pixel.color)]

    @staticmethod
    def get_similarity_gray_pixels(src_pixels: [SeedPixel], parent_reference_color: QColor, color_diff) -> []:
        """
        :param src_pixels:
        :param parent_reference_color:
        :param color_diff:
        :return:
        """

        def colour_distance1(rgb_1: QColor, rgb_2: QColor):
            r_1, g_1, b_1 = rgb_1.red(), rgb_1.green(), rgb_1.blue()
            r_2, g_2, b_2 = rgb_2.red(), rgb_2.green(), rgb_2.blue()
            rmean = (r_1 + r_2) / 2
            r = r_1 - r_2
            g = g_1 - g_2
            b = b_1 - b_2
            return sqrt((2 + rmean / 256) * (r ** 2) + 4 * (g ** 2) + (2 + (255 - rmean) / 256) * (b ** 2))

        def colour_distance2(rgb_1: QColor, rgb_2: QColor):
            r_1, g_1, b_1 = rgb_1.red(), rgb_1.green(), rgb_1.blue()
            r_2, g_2, b_2 = rgb_2.red(), rgb_2.green(), rgb_2.blue()
            return sqrt((r_1 - r_2) ** 2 + (g_1 - g_2) ** 2 + (b_1 - b_2) ** 2)

        result = []
        # 颜色相似度计算方法1
        if color_diff:
            result = [pixel for pixel in src_pixels if
                      colour_distance2(pixel.color, parent_reference_color) <= color_diff]
        return result


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
    for y in range(position.y() - radius, position.y() + radius + 1):
        for x in range(position.x() - radius, position.x() + radius + 1):
            if (pow(x - position.x(), 2) + pow(y - position.y(), 2)) <= pow(radius, 2):
                pixel_color = image.pixelColor(x, y)
                result.append(SeedPixel(QPoint(x, y), pixel_color))
    return result


def calculate_reference_color_of(circle_seed) -> QColor:
    seed_pixels = circle_seed.gray_pixels
    if not seed_pixels:
        return GrayPixelsSet.standard_gray()

    sum_r, sum_g, sum_b = 0, 0, 0
    for seed_pixel in seed_pixels:
        sum_r += seed_pixel.r()
        sum_g += seed_pixel.g()
        sum_b += seed_pixel.b()

    m = len(seed_pixels)
    return QColor(int(sum_r / m), int(sum_g / m), int(sum_b / m))


def is_accept_color_diff(color: QColor, color_diff):
    r, g, b = color.red(), color.green(), color.blue()
    res = abs(r - g), abs(r - b), abs(g - b) <= color_diff
    return all(res)


def get_circle_seed_path(circle_seed) -> QPainterPath:
    rect = QRectF(circle_seed.position.x() - circle_seed.radius, circle_seed.position.y() - circle_seed.radius,
                  circle_seed.radius * 2, circle_seed.radius * 2)
    seed_path = QPainterPath()
    seed_path.arcMoveTo(rect, 0)
    seed_path.arcTo(rect, 0, 360)
    seed_path.closeSubpath()
    return seed_path


def get_circle_path(center_pos: list, radius: int):
    rect = QRectF(center_pos[0] - radius, center_pos[1] - radius, radius * 2, radius * 2)
    seed_path = QPainterPath()
    seed_path.arcMoveTo(rect, 0)
    seed_path.arcTo(rect, 0, 360)
    seed_path.closeSubpath()
    return seed_path


def adjust_angle(src_angle: float) -> float:
    while src_angle < 0.:
        src_angle = 2 * pi + src_angle

    while src_angle > 2 * pi:
        src_angle = src_angle - 2 * pi

    return src_angle


def calculate_spectral_distance(spectral_info_vector1, spectral_info_vector2):
    return np.linalg.norm(np.subtract(spectral_info_vector1, spectral_info_vector2))


def create_circle_path(x_center: int, y_center: int, radius: int) -> QPainterPath:
    rect = QRectF(x_center - radius, y_center - radius, radius * 2, radius * 2)
    circle_path = QPainterPath()
    circle_path.arcMoveTo(rect, 0)
    circle_path.arcTo(rect, 0, 360)
    circle_path.closeSubpath()
    return circle_path


def get_pixels_from_path(path: QPainterPath) -> list:
    polygons = path.toFillPolygons()
    result = []
    for polygon in polygons:
        for i in range(polygon.size()):
            result.append(polygon.at(i))
    return result


def bound(min_val, value, max_val):
    """
    如果value >= max_val return max_val
    如果value <= min_val return min_val
    否则 return value

    :param min_val: 最小值
    :param value: 被测值
    :param max_val: 最大值
    """

    return max(min_val, min(max_val, value))


def combination22(src_list: [tuple, list]) -> list:
    result = []
    list_len = len(src_list)
    for i in range(0, list_len):
        for j in range(i + 1, list_len):
            result.append((src_list[i], src_list[j]))
    return result


def qimage2cvmat(image: QImage, image_format=QImage.Format_RGB32, channel=4) -> np.ndarray:
    """Converts a QImage into an opencv MAT format"""

    incoming_image = image.convertToFormat(image_format)

    width = incoming_image.width()
    height = incoming_image.height()

    ptr = incoming_image.bits()
    ptr.setsize(incoming_image.byteCount())
    arr = np.array(ptr).reshape((height, width, channel))  # Copies the data
    return arr


def cvmat2qimage(cv_image: np.ndarray):
    shape = cv_image.shape  # type: tuple
    if len(shape) == 2:
        image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)  # type: np.ndarray
    elif shape[2] == 4:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)  # type: np.ndarray
    else:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)   # type: np.ndarray
    return QImage(image.tobytes(), image.shape[0], image.shape[1], QImage.Format_RGB888)


if __name__ == '__main__':

    class M:
        def __init__(self, m):
            self.m = m

        def __lt__(self, other):
            print("call lt")
            return self.m < other.m

        def __gt__(self, other):
            print("call gt")
            return self.m > other.m

        def __eq__(self, other):
            print("call eq")
            return self.m == other.m

        def __str__(self):
            print("call str: ", end=" ")
            return str(self.m)

    m1 = M(4)
    m2 = M(3)
    m = [m1, m2, M(2), M(10)]
    n = sorted(m)

    for i in m:
        print(i)

    for i in n:
        print(i)
