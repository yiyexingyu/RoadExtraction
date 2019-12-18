# -*- coding: utf-8 -*-
# @Time    : 2019/10/3 20:42
# @Author  : 小羽
# @Email   : 2958029539@qq.com
# @File    : Utils.py
# @Project : RoadExtraction
# @Software: PyCharm

from math import pi
import numpy as np
import cv2
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QImage, QPainterPath


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


def adjust_angle(src_angle: [float, np.ndarray]) -> float:
    if isinstance(src_angle, float):
        while src_angle < 0.:
            src_angle = 2 * pi + src_angle

        while src_angle > 2 * pi:
            src_angle = src_angle - 2 * pi
    elif isinstance(src_angle, np.ndarray):
        neg_angle = src_angle[src_angle < 0]   # type: np.ndarray
        while neg_angle.size > 0:
            neg_angle += 2 * pi
            src_angle[src_angle < 0] = neg_angle
            neg_angle = src_angle[src_angle < 0]  # type: np.ndarray

        large_angle = src_angle[src_angle > 2 * pi]  # type: np.ndarray
        while large_angle.size > 0:
            large_angle -= 2 * pi
            src_angle[src_angle > 2 * pi] = large_angle
            large_angle = src_angle[src_angle > 2 * pi]  # type: np.ndarray

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
    :type min_val: Union[int, float]
    :param value: 被测值
    :type value: Union[int, float]
    :param max_val: 最大值
    :type max_val: Union[int, float]
    :rtype: float
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
