# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 16:25
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : TestDirection.py
# @Project : RoadExtraction
# @Software: PyCharm

import math
from math import pi, cos, sin
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import pyqtSignal, QPoint, Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap, QPen
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import get_pixels_from
from Test.CircleSeedItem import CircleSeedItem
from Core.DetectionAlgorithm.GeneralSimilarityDetection import general_similarity_detection


class ShowResultLabel(QLabel):

    def __init__(self, parent):
        super(ShowResultLabel, self).__init__(parent)
        self._circle_sed_item_list = []

    def add_circle_seed(self, circle_seed: CircleSeed):
        self._circle_sed_item_list.append(CircleSeedItem(circle_seed.position, circle_seed.radius))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        for item in self._circle_sed_item_list:  # type: CircleSeedItem
            print(item.center_pos)
            painter.drawPath(item.path)
            painter.drawLine(self._circle_sed_item_list[0].center_pos, item.center_pos)
            # painter.fillPath(item.path, item.color())
        # event.accept()


class TestDirection(QWidget):

    def __init__(self, image: QImage):
        super(TestDirection, self).__init__()
        self.setWindowTitle("道路提取算法测试")
        self.setFixedWidth(image.width())
        self.setFixedHeight(image.height())
        self._image_label = ShowResultLabel(self)
        self._image = image
        position = QPoint(700, 350)
        pixels = get_pixels_from(image, position, 33)
        self._circle_seed = CircleSeed(position, 33, pixels, direction=0)

        self._angle_interval = pi / 12
        self._current_angle_k = 0
        self._image_label.add_circle_seed(self._circle_seed)
        self._image_label.setPixmap(QPixmap.fromImage(image))

    def next_circle_seed(self, parent_circle_seed: CircleSeed) -> CircleSeed:
        current_angle = self._current_angle_k * self._angle_interval + parent_circle_seed.direction
        if current_angle > 2 * pi:
            current_angle = current_angle % (2 * pi)

        moving_distance = parent_circle_seed.radius * 2
        x_candidate = parent_circle_seed.position.x() + int(moving_distance * cos(current_angle))
        y_candidate = parent_circle_seed.position.y() - int(moving_distance * sin(current_angle))
        current_pos = QPoint(x_candidate, y_candidate)

        # 计算出候选种子的像素集
        seed_pixels = get_pixels_from(self._image, current_pos, parent_circle_seed.radius)
        # 创建候选种子
        candidate_seed = CircleSeed(current_pos, parent_circle_seed.radius, seed_pixels, current_angle)
        self._current_angle_k += 1
        return candidate_seed

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_A:
            self._image_label.add_circle_seed(self.next_circle_seed(self._circle_seed))


if __name__ == '__main__':
    import sys
    app = QApplication([])

    image_path = "F:/RoadDetectionTestImg/4.png"
    image1 = QImage(image_path)

    win = TestDirection(image1)
    win.show()

    sys.exit(app.exec_())
