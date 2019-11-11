# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 17:15
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : ShowResultView.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5 import QtGui
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter

from DetectObjects.CircleSeed import CircleSeedNp as CircleSeed
from Test.ShowResultLabel import ShowResultItem


class ShowResultView(QGraphicsView):

    circle_seed_clicked_signal = pyqtSignal(int, CircleSeed)
    start_road_detection_signal = pyqtSignal(CircleSeed)

    def __init__(self, main_item: ShowResultItem = None, parent=None):
        super(ShowResultView, self).__init__(parent)

        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._main_item = main_item

    @property
    def main_item(self):
        return self._main_item

    @main_item.setter
    def main_item(self, new_item):
        self._main_item = new_item

    def add_circle_seed(self, circle_seed: CircleSeed):
        self._circle_seeds_list.append(circle_seed)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self._main_item.keyPressEvent(event)

    def keyReleaseEvent(self, event0: QtGui.QKeyEvent) -> None:
        self._main_item.keyReleaseEvent(event0)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self._main_item.wheelEvent(event)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._main_item.mousePressEvent(ev)
        self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._main_item.mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        self._main_item.mouseMoveEvent(ev)
        self.update()

    # def paintEvent(self, event: QtGui.QPaintEvent) -> None:
    #     super().paintEvent(event)
    #     painter = QPainter(self)
    #     painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
    #
    #     print(len(self._circle_seed_items))
    #     for index, circle_seed_item in enumerate(self._circle_seed_items):  # type: int, CircleSeedItem
    #
    #         # if index == self._current_item_index and circle_seed_item.change_able:
    #         #     painter.setPen(Qt.black)
    #         #     painter.fillRect(circle_seed_item.resize_handel().adjusted(0, 0, -1, -1), Qt.black)
    #         print("draw")
    #         path = self.mapToScene(circle_seed_item.path)
    #         if index == self._current_item_index:
    #             painter.fillPath(path, Qt.green)
    #         else:
    #             painter.fillPath(path, Qt.red)
