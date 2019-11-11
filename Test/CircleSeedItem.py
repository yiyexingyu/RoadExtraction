# -*- coding: utf-8 -*-
# @Time    : 2019/10/20 23:10
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : CircleSeedItem.py
# @Project : RoadExtraction
# @Software: PyCharm

from enum import IntEnum
import numpy as np
from PyQt5.QtGui import QColor, QPainterPath, QImage
from PyQt5.QtCore import QSizeF, QRect, QPoint, Qt, QRectF, pyqtSignal, QObject


RESIZE_HANDEL_WIDTH = 6


class Type(IntEnum):
    Rectangle = 1
    Circle = 2
    Triangle = 3


class CircleSeedItem(QObject):

    position_changed_signal = pyqtSignal(list, np.ndarray)
    radius_changed_signal = pyqtSignal(int, np.ndarray)
    min_size = QSizeF(10, 10)

    def __init__(self, cv_image: np.ndarray, center_pos: list, radius,
                 shape_type=Type.Circle, color: QColor = Qt.red, can_change=False):
        super(CircleSeedItem, self).__init__()
        self.__type = shape_type
        # self.__path = path
        # self._image = image
        self._cv_image = cv_image
        self.__center_pos = center_pos
        self.__radius = radius
        self.__color = color
        self.__change_able = can_change
        self.__name = CircleSeedItem.type_to_string(shape_type)

    def type(self) -> Type:
        return self.__type

    def rect(self) -> QRect:
        return QRect(self.__center_pos[0] - self.__radius, self.__center_pos[1] - self.__radius,
                     self.__radius * 2, self.__radius * 2)

    def name(self) -> str:
        return self.__name

    def set_name(self, name):
        self.__name = name

    def set_color(self, color: QColor):
        self.__color = color

    def resize_handel(self) -> QRect:
        br = self.rect().bottomRight()
        return QRect((br - QPoint(RESIZE_HANDEL_WIDTH, RESIZE_HANDEL_WIDTH)), br)

    def color(self) -> QColor:
        return self.__color

    @property
    def path(self):
        path = QPainterPath()
        rect = QRectF(self.rect())
        path.arcMoveTo(rect, 0)
        path.arcTo(rect, 0, 360)
        path.closeSubpath()
        return path

    def get_path(self):
        path = QPainterPath()
        rect = QRectF(self.rect())
        path.arcMoveTo(rect, 0)
        path.arcTo(rect, 0, 360)
        path.closeSubpath()
        return path

    @property
    def radius(self):
        return self.__radius

    @radius.setter
    def radius(self, new_radius):
        self.__radius = new_radius if new_radius > 5 else 5
        self.radius_changed_signal.emit(new_radius, self._cv_image)

    @property
    def center_pos(self) -> list:
        return self.__center_pos

    @center_pos.setter
    def center_pos(self, new_pos):
        self.__center_pos = new_pos
        self.position_changed_signal.emit(new_pos, self._cv_image)

    @property
    def change_able(self):
        return self.__change_able

    @change_able.setter
    def change_able(self, change_able_):
        self.__change_able = change_able_

    @staticmethod
    def type_to_string(type_shape: Type) -> [str, None]:
        if not isinstance(type_shape, Type):
            return None
        return type_shape.name

    @staticmethod
    def string_to_type(shape_string: str) -> (Type, bool):
        type_dict = dict(Type.__members__)
        try:
            return type_dict[shape_string], True
        except KeyError:
            return Type.Rectangle, False
        finally:
            return Type.Rectangle, False

