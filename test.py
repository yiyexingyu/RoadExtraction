# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 22:10
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : test.py
# @Project : RoadExtraction
# @Software: PyCharm

import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QPoint, Qt, QRectF, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPainter, QPen, QPainterPath, QPixmap
from Core.RoadDetection import RoadDetection, CircleSeed, GNSDetectionStrategy
from Core.DetectionStrategy.AbstractDetectionStrategy import AbstractDetectionStrategy
from DetectObjects.Utils import get_circle_seed_path


class ShowResultLabel(QLabel):

    def __init__(self, parent):
        super(ShowResultLabel, self).__init__(parent)
        self._circle_seeds_list = []

    def new_seed_generated(self, child_seed):
        self._circle_seeds_list.append(child_seed)
        self.update()

    def add_circle_seed(self, circle_seed: CircleSeed):
        self._circle_seeds_list.append(circle_seed)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        for circle_seed in self._circle_seeds_list:
            seed_path = get_circle_seed_path(circle_seed)
            painter.fillPath(seed_path, Qt.red)
            # painter.drawPath(seed_path)


class ShowImageWidget(QWidget):

    new_seeds_generated = pyqtSignal(CircleSeed)

    def __init__(self, image: QImage):
        super(ShowImageWidget, self).__init__()
        self.setWindowTitle("道路提取算法测试")
        self.setFixedWidth(image.width())
        self.setFixedHeight(image.height())

        self._image_label = ShowResultLabel(self)
        self._image_label.setGeometry(0, 0, image.width(), image.height())
        self.new_seeds_generated.connect(self._image_label.new_seed_generated)
        self._image_label.setPixmap(QPixmap.fromImage(image))

    def add_circle_seed(self, circle_seed: CircleSeed):
        self._image_label.add_circle_seed(circle_seed)


class RoadDetectThread(QThread):

    road_detect_finished_signal = pyqtSignal(RoadDetection)

    def __init__(self, road_detection: RoadDetection):
        super(RoadDetectThread, self).__init__()
        self._road_detection = road_detection

    def run(self) -> None:
        self._road_detection.road_detection()
        print("线程任务完成！")
        self.road_detect_finished_signal.emit(self._road_detection)


def show_generated_seeds_info(generated_seed: CircleSeed):
    print("========================================================")
    print("子种子：")
    print(generated_seed)
    print("========================================================")


def road_detect_finished(road_detection: RoadDetection):
    print("道路检测结束：")
    print("共产生圆形种子：", (len(road_detection.get_seed_list()) - 1), " 个")


if __name__ == '__main__':
    image_path = "F:/RoadDetectionTestImg/4.png"
    image1 = QImage(image_path)
    position = QPoint(500, 433)
    radius = 11

    road_detect = RoadDetection(image1)
    road_detect.circle_seeds_generated.connect(show_generated_seeds_info)
    circle_seed1 = road_detect.initialize(position, radius)

    app = QApplication(sys.argv)
    window = ShowImageWidget(image1)
    window.add_circle_seed(circle_seed1)
    road_detect.circle_seeds_generated.connect(window.new_seeds_generated)

    road_detect_thread = RoadDetectThread(road_detect)
    road_detect_thread.road_detect_finished_signal.connect(road_detect_finished)

    window.show()
    print("开始进行道路检测....")
    road_detect_thread.start()

    sys.exit(app.exec_())
