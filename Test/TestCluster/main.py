# -*- encoding: utf-8 -*-
# @File    : main.py
# @Time    : 2019/11/27 12:40
# @Author  : 一叶星羽
# @Email   : h0670131005@gmail.com
# @Software: PyCharm

import sys
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication

from PyQt5.QtGui import QImage, QPainter, QColor
from DetectObjects.CircleSeed import CircleSeedNp
from Core.DetectionStrategy.Strategy import DetectionStrategy
from Core.Cluster.KMeanCluster import k_mean_cluster


class Window(QWidget):

    def __init__(self, image_path: str):
        super(Window, self).__init__()

        win_rectangle = self.frameGeometry()  # 这是得到窗口矩形
        center_point = QDesktopWidget().availableGeometry().center()  # 获取屏幕的中心点坐标。
        win_rectangle.moveCenter(center_point)  # 将窗口同型矩形winRectangle，移动到屏幕中央
        self.move(win_rectangle.topLeft())  # 将窗口移动到那个同型矩形winRectangle中

        self._current_circle = None
        self._init_seeds = []
        self._cv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self._image = QImage(image_path)
        self.setFixedSize(self._image.size())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            if self._current_circle:
                self._init_seeds.remove(self._current_circle)
                self.update()
        elif event.key() == Qt.Key_S:
            thread = DetectThread(self._init_seeds, self._cv_image, self)
            thread.start()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()
        pos = [event.pos().x(), event.pos().y()]

        for circle_seed in self._init_seeds:  # type: CircleSeedNp
            if circle_seed.road_path.contains(event.pos()):
                self._current_circle = circle_seed
                self.update()
                return

        self._init_seeds.append(CircleSeedNp(pos, 5, DetectionStrategy.Initialization, image=self._cv_image))
        self._current_circle = self._init_seeds[-1]
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        event.accept()
        if isinstance(self._current_circle, CircleSeedNp):
            self._current_circle.set_radius(
                self._current_circle.radius + (1 if event.angleDelta().y() > 0 else -1), self._cv_image)
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:

        painter = QPainter(self)
        painter.drawImage(self._image.rect(), self._image)
        for circle_seed in self._init_seeds:  # type: CircleSeedNp
            if self._current_circle and circle_seed == self._current_circle:
                painter.fillPath(circle_seed.road_path, QColor(0, 255, 0, 200))
            else:
                painter.fillPath(circle_seed.road_path, QColor(255, 0, 0, 200))
        super().paintEvent(event)


class DetectThread(QThread):

    def __init__(self, init_seeds: list, cv_image, parent=None):
        super(DetectThread, self).__init__(parent)
        self._road_feature = DetectThread._road_feature_sample(init_seeds)
        self._cv_image = cv_image

    @staticmethod
    def _road_feature_sample(init_seeds):
        if init_seeds is None or len(init_seeds) == 0:
            return None
        road_feature_samples = []
        for circle_seed in init_seeds:  # type: CircleSeedNp
            road_feature_samples.append(circle_seed.road_feature_vector)

        road_feature_samples = np.array(road_feature_samples)
        return np.apply_over_axes(np.mean, road_feature_samples, 0)[0]

    def run(self) -> None:
        print("开始检测...")
        res_image = k_mean_cluster(self._cv_image, self._road_feature)
        cv2.imshow("cluster result", res_image)
        cv2.waitKey(0)
        cv2.destroyWindow("image")


def application():
    app = QApplication([])

    image_path = "../../TestImg/5.png"
    win = Window(image_path)
    win.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    application()
