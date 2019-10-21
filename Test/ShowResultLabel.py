# -*- coding: utf-8 -*-
# @Time    : 2019/10/20 23:20
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : ShowResultLabel.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QPen, QPainter

from DetectObjects.CircleSeed import CircleSeed
from .CircleSeedItem import CircleSeedItem


class ShowResultLabel(QLabel):

    circle_seed_clicked_signal = pyqtSignal(int, CircleSeed)
    start_road_detection_signal = pyqtSignal(CircleSeed)

    def __init__(self, parent):
        super(ShowResultLabel, self).__init__(parent)
        self._is_road_detecting = False
        self._has_init_circle_seed = False
        self._resize_handel_pressed = False
        self._mouse_press_offset = QPoint()
        self._current_item_index = -1
        self._circle_seed_items = []
        self._circle_seeds_list = []

    @property
    def is_road_detecting(self):
        return self._is_road_detecting

    @is_road_detecting.setter
    def is_road_detecting(self, road_detecting):
        self._is_road_detecting = road_detecting

    def road_detection_finished(self, ignore):
        self._is_road_detecting = False
        self._has_init_circle_seed = False

    def new_seed_generated(self, child_seed: CircleSeed):
        self._circle_seeds_list.append(child_seed)
        self._circle_seed_items.append(
            CircleSeedItem(child_seed.position, child_seed.radius))
        self.update()

    def add_circle_seed(self, circle_seed: CircleSeed):
        self._circle_seeds_list.append(circle_seed)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        event.accept()
        if event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            if not self._is_road_detecting and self._has_init_circle_seed:
                self._is_road_detecting = True
                init_circle_seed_item = self._circle_seed_items[0]   # type: CircleSeedItem
                init_circle_seed_item.change_able = False
                self.start_road_detection_signal.emit(self._circle_seeds_list[0])

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        event.accept()
        if self._current_item_index != -1:
            current_seed_item = self._circle_seed_items[self._current_item_index]  # type: CircleSeedItem
            if current_seed_item.change_able:
                current_seed_item.radius += (1 if event.angleDelta().y() > 0 else -1)
                self.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        ev.accept()
        current_seed_item = None
        for seed_item in self._circle_seed_items:  # type: CircleSeedItem
            if seed_item.path.contains(ev.pos()):
                current_seed_item = seed_item
                break

        if current_seed_item:  # type: CircleSeedItem
            self._current_item_index = self._circle_seed_items.index(current_seed_item)
            self.circle_seed_clicked_signal.emit(
                self._current_item_index, self._circle_seeds_list[self._current_item_index])
            if current_seed_item.change_able:
                self._resize_handel_pressed = current_seed_item.resize_handel().contains(ev.pos())
                if self._resize_handel_pressed:
                    self._mouse_press_offset = ev.pos() - current_seed_item.rect().bottomRight()
                else:
                    self._mouse_press_offset = ev.pos() - current_seed_item.rect().topLeft()
        elif not self._has_init_circle_seed:
            init_circle_seed_item = CircleSeedItem(ev.pos(), 11, can_change=True)
            init_circle_seed = CircleSeed(init_circle_seed_item.center_pos, init_circle_seed_item.radius, [])
            self._circle_seed_items.append(init_circle_seed_item)
            self._circle_seeds_list.append(init_circle_seed)
            init_circle_seed_item.position_changed_signal.connect(init_circle_seed.set_position)
            init_circle_seed_item.radius_changed_signal.connect(init_circle_seed.set_radius)

            self._has_init_circle_seed = True
            self._current_item_index = 0
        else:
            self._current_item_index = -1
        self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        ev.accept()
        self._resize_handel_pressed = False
        # self._current_item_index = -1

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        ev.accept()
        if self._current_item_index == -1:
            return

        current_seed_item = self._circle_seed_items[self._current_item_index]  # type: CircleSeedItem
        if current_seed_item.change_able:
            if self._resize_handel_pressed:
                rect = QRect(current_seed_item.rect().topLeft(), QPoint(ev.pos() + self._mouse_press_offset))
                current_seed_item.radius = min(rect.width(), rect.height(), 8)
            else:
                rect = current_seed_item.rect()
                rect.moveTopLeft(ev.pos() - self._mouse_press_offset)
                rect = rect.adjusted(0, 0, 1, 1)
                current_seed_item.center_pos = rect.center()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        for index, circle_seed_item in enumerate(self._circle_seed_items):  # type: int, CircleSeedItem
            painter.fillPath(circle_seed_item.path, Qt.red)
            if index == self._current_item_index and circle_seed_item.change_able:
                painter.setPen(Qt.black)
                painter.fillRect(circle_seed_item.resize_handel().adjusted(0, 0, -1, -1), Qt.black)
