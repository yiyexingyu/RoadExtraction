# -*- coding: utf-8 -*-
# @Time    : 2019/10/20 23:20
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : ShowResultLabel.py
# @Project : RoadExtraction
# @Software: PyCharm

import typing
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QMenu, QAction, QGraphicsObject, QWidget, QStyleOptionGraphicsItem
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QPen, QPainter, QImage

from DetectObjects.CircleSeed import CircleSeedNp
from .CircleSeedItem import CircleSeedItem
from .CircleSeedDetail import CircleSeedDetail
from Core.RoadDetection import RoadDetection
from Core.DetectionStrategy.Strategy import DetectionStrategy
from .OpenCVAnalysis import show_analysis_info, compare_to_seed, show_grabcut_info, compare_tow_seed_of_spectral_info


class ShowResultLabel(QLabel):

    circle_seed_clicked_signal = pyqtSignal(int, CircleSeedNp)
    start_road_detection_signal = pyqtSignal(CircleSeedNp)

    def __init__(self, image: QImage, cv_image: np.ndarray, road_detection: RoadDetection, parent):
        super(ShowResultLabel, self).__init__(parent)
        self._is_road_detecting = False
        self._has_init_circle_seed = False
        self._resize_handel_pressed = False
        self._mouse_press_offset = QPoint()
        self._current_item_index = -1
        self._circle_seed_items = []
        self._circle_seeds_list = []
        self._context_menu = self.__init_context_menu()
        self._image = image
        self._cv_image = cv_image
        self._road_detection = road_detection

        self._select_items_flag = False
        self._temp_seed = None
        self._selected_indexes = []

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def __init_context_menu(self) -> QMenu:
        menu = QMenu(self)

        delete_action = QAction("删除该种子", menu)
        delete_action.triggered.connect(self.delete_seed)

        show_seed_detail_action = QAction("详细信息", menu)
        show_seed_detail_action.triggered.connect(self.show_seed_detail)

        opencv_analysis_menu = menu.addMenu("用opencv分析")
        opencv_analysis_action = QAction("阈值法", opencv_analysis_menu)
        opencv_grabcut_action = QAction("GrabCut", opencv_analysis_menu)
        opencv_analysis_action.triggered.connect(self.analysis_with_openCV)
        opencv_grabcut_action.triggered.connect(self.grabcut_opencv)

        opencv_analysis_menu.addAction(opencv_analysis_action)
        opencv_analysis_menu.addAction(opencv_grabcut_action)

        menu.addAction(delete_action)
        menu.addAction(show_seed_detail_action)
        # menu.addAction(opencv_analysis_action)

        return menu

    def grabcut_opencv(self):
        if self._current_item_index != -1:
            show_grabcut_info(self._image, self._circle_seed_items[self._current_item_index])

    def analysis_with_openCV(self):
        if self._current_item_index != -1:
            show_analysis_info(self._image, self._circle_seed_items[self._current_item_index])

    def show_seed_detail(self):
        if self._current_item_index != -1:
            csd = CircleSeedDetail(self.parent(), self._circle_seeds_list[self._current_item_index])
            csd.show()

    def delete_seed(self):
        if self._current_item_index != -1:
            self._circle_seed_items.pop(self._current_item_index)
            self._circle_seeds_list.pop(self._current_item_index)
            self._current_item_index = -1
            if len(self._circle_seeds_list) == 0:
                self._has_init_circle_seed = False
            self.update()

    def show_context_menu(self, pos):
        if self._current_item_index != -1:
            self._context_menu.exec_(self.mapToGlobal(pos))

    @property
    def is_road_detecting(self):
        return self._is_road_detecting

    @is_road_detecting.setter
    def is_road_detecting(self, road_detecting):
        self._is_road_detecting = road_detecting

    def road_detection_finished(self, ignore):
        self._is_road_detecting = False
        self._has_init_circle_seed = False

    def new_seed_generated(self, child_seed: CircleSeedNp):
        self._circle_seeds_list.append(child_seed)
        self._circle_seed_items.append(
            CircleSeedItem(self._cv_image, child_seed.position, child_seed.radius))
        self._current_item_index = len(self._circle_seed_items) - 1
        self.update()

    def add_circle_seed(self, circle_seed: CircleSeedNp):
        self._circle_seeds_list.append(circle_seed)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        event.accept()
        if event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            if not self._is_road_detecting and self._has_init_circle_seed:
                self._is_road_detecting = True
                init_circle_seed_item = self._circle_seed_items[0]   # type: CircleSeedItem
                init_circle_seed_item.change_able = False
                self.start_road_detection_signal.emit(self._circle_seeds_list[0])
        elif event.key() == Qt.Key_A:
            self._select_items_flag = True

    def keyReleaseEvent(self, event0: QtGui.QKeyEvent) -> None:
        event0.accept()
        if event0.key() == Qt.Key_A:
            self._select_items_flag = False
            if self._temp_seed:
                compare_tow_seed_of_spectral_info(self._cv_image, self._circle_seeds_list[self._current_item_index], self._temp_seed)
                self._selected_indexes = self._road_detection.validation_road_pixels_proportion(self._circle_seeds_list[self._current_item_index])
                self._temp_seed = None

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

        if current_seed_item and self._select_items_flag and self._current_item_index != -1:
            self._temp_seed = self._circle_seeds_list[self._circle_seed_items.index(current_seed_item)]
            return

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
            init_circle_seed_item = CircleSeedItem(self._cv_image, [ev.pos().x(), ev.pos().y()], 11, can_change=True)
            init_circle_seed = CircleSeedNp(init_circle_seed_item.center_pos, init_circle_seed_item.radius,
                                            DetectionStrategy.Initialization, image=self._cv_image)
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
                center = rect.adjusted(0, 0, 1, 1).center()
                current_seed_item.center_pos = [center.x(), center.y()]
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        for index, circle_seed_item in enumerate(self._circle_seed_items):  # type: int, CircleSeedItem

            if index == self._current_item_index and circle_seed_item.change_able:
                painter.setPen(Qt.black)
                painter.fillRect(circle_seed_item.resize_handel().adjusted(0, 0, -1, -1), Qt.black)
            if index == self._current_item_index or index in self._selected_indexes:
                try:
                    painter.fillPath(circle_seed_item.get_path(), Qt.green)
                except Exception as e:
                    print(e)
            else:
                painter.fillPath(circle_seed_item.get_path(), Qt.red)


class ShowResultItem(QGraphicsObject):

    circle_seed_clicked_signal = pyqtSignal(int, CircleSeedNp)
    start_road_detection_signal = pyqtSignal(CircleSeedNp)
    init_circle_seed_signal = pyqtSignal(bool)

    def __init__(self, image: QImage = None, parent=None):
        super(ShowResultItem, self).__init__(parent)
        self._is_road_detecting = False
        self._has_init_circle_seed = False
        self._resize_handel_pressed = False
        self._mouse_press_offset = QPoint()
        self._current_item_index = -1
        self._circle_seed_items = []
        self._circle_seeds_list = []
        self._context_menu = self.__init_context_menu()
        self._image = image

        self._select_items_flag = False
        self._temp_seed = None

    @property
    def image(self) -> QImage:
        return self._image

    @image.setter
    def image(self, new_image: QImage):
        self._image = new_image
        self.update(QtCore.QRectF(self._image.rect()))

    def __init_context_menu(self) -> QMenu:
        menu = QMenu(self.parentWidget())

        delete_action = QAction("删除该种子", menu)
        delete_action.triggered.connect(self.delete_seed)

        show_seed_detail_action = QAction("详细信息", menu)
        show_seed_detail_action.triggered.connect(self.show_seed_detail)

        opencv_analysis_menu = menu.addMenu("用opencv分析")
        opencv_analysis_action = QAction("阈值法", opencv_analysis_menu)
        opencv_grabcut_action = QAction("GrabCut", opencv_analysis_menu)
        opencv_analysis_action.triggered.connect(self.analysis_with_openCV)
        opencv_grabcut_action.triggered.connect(self.grabcut_opencv)

        opencv_analysis_menu.addAction(opencv_analysis_action)
        opencv_analysis_menu.addAction(opencv_grabcut_action)

        menu.addAction(delete_action)
        menu.addAction(show_seed_detail_action)
        # menu.addAction(opencv_analysis_action)

        return menu

    def grabcut_opencv(self):
        if self._current_item_index != -1:
            show_grabcut_info(self._image, self._circle_seed_items[self._current_item_index])

    def analysis_with_openCV(self):
        if self._current_item_index != -1:
            show_analysis_info(self._image, self._circle_seed_items[self._current_item_index])

    def show_seed_detail(self):
        if self._current_item_index != -1:
            csd = CircleSeedDetail(self.parent(), self._circle_seeds_list[self._current_item_index])
            csd.show()

    def delete_seed(self):
        if self._current_item_index != -1:
            self._circle_seed_items.pop(self._current_item_index)
            self._circle_seeds_list.pop(self._current_item_index)
            self._current_item_index = -1
            if len(self._circle_seeds_list) == 0:
                self._has_init_circle_seed = False
            self.update()

    def contextMenuEvent(self, event: 'QGraphicsSceneContextMenuEvent') -> None:
        pos = event.pos()
        if self._current_item_index != -1:
            self._context_menu.exec_(self.mapToGlobal(pos))

    @property
    def is_road_detecting(self):
        return self._is_road_detecting

    @is_road_detecting.setter
    def is_road_detecting(self, road_detecting):
        self._is_road_detecting = road_detecting

    def road_detection_finished(self, ignore):
        self._is_road_detecting = False
        self._has_init_circle_seed = False

    def new_seed_generated(self, child_seed: CircleSeedNp):
        self._circle_seeds_list.append(child_seed)
        self._circle_seed_items.append(
            CircleSeedItem(self._image, child_seed.position, child_seed.radius))
        self._current_item_index = len(self._circle_seed_items) - 1
        self.update()

    def add_circle_seed(self, circle_seed: CircleSeedNp):
        self._circle_seeds_list.append(circle_seed)

    def boundingRect(self) -> QtCore.QRectF:
        if self._image:
            return QtCore.QRectF(self._image.rect())
        else:
            return QtCore.QRectF(0, 0, 100, 100)

    def start_road_detect(self) -> CircleSeedNp:
        if not self._is_road_detecting and self._has_init_circle_seed:
            self._is_road_detecting = True
            # self._has_init_circle_seed = False
            init_circle_seed_item = self._circle_seed_items[-1]  # type: CircleSeedItem
            init_circle_seed_item.change_able = False
            return self._circle_seeds_list[-1]
        else:
            return None

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        event.accept()
        # if event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
        #     self.start_road_detect()
        if event.key() == Qt.Key_A:
            self._select_items_flag = True

    def keyReleaseEvent(self, event0: QtGui.QKeyEvent) -> None:
        event0.accept()
        if event0.key() == Qt.Key_A:
            self._select_items_flag = False
            if self._temp_seed:
                compare_to_seed(self._image, self._circle_seed_items[self._current_item_index], self._temp_seed)
                self._temp_seed = None

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
        pos = self.mapToScene(ev.pos()).toPoint()
        for seed_item in self._circle_seed_items:  # type: CircleSeedItem
            if seed_item.path.contains(pos):
                current_seed_item = seed_item
                break

        if current_seed_item and self._select_items_flag and self._current_item_index != -1:
            self._temp_seed = current_seed_item
            return

        if current_seed_item:  # type: CircleSeedItem
            self._current_item_index = self._circle_seed_items.index(current_seed_item)
            self.circle_seed_clicked_signal.emit(
                self._current_item_index, self._circle_seeds_list[self._current_item_index])
            if current_seed_item.change_able:
                self._resize_handel_pressed = current_seed_item.resize_handel().contains(pos)
                if self._resize_handel_pressed:
                    self._mouse_press_offset = pos - current_seed_item.rect().bottomRight()
                else:
                    self._mouse_press_offset = pos - current_seed_item.rect().topLeft()
        elif not self._has_init_circle_seed:
            init_circle_seed_item = CircleSeedItem(self._image, pos, 11, can_change=True)
            init_circle_seed = CircleSeedNp(init_circle_seed_item.center_pos, init_circle_seed_item.radius, [])
            self._circle_seed_items.append(init_circle_seed_item)
            self._circle_seeds_list.append(init_circle_seed)
            init_circle_seed_item.position_changed_signal.connect(init_circle_seed.set_position)
            init_circle_seed_item.radius_changed_signal.connect(init_circle_seed.set_radius)

            self._has_init_circle_seed = True
            self._current_item_index = 0
            self.init_circle_seed_signal.emit(True)
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

    def paint(self, painter: QtGui.QPainter, option: 'QStyleOptionGraphicsItem', widget: typing.Optional[QWidget] = ...) -> None:
        # super().paintEvent(event)
        # painter = QPainter(self)
        if self._image:
            rect = self.mapRectToScene(QtCore.QRectF(0, 0, self._image.width() - 1, self._image.height() - 1))
            painter.drawImage(rect, self._image)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))

        for index, circle_seed_item in enumerate(self._circle_seed_items):  # type: int, CircleSeedItem

            # if index == self._current_item_index and circle_seed_item.change_able:
            #     painter.setPen(Qt.black)
            #     painter.fillRect(circle_seed_item.resize_handel().adjusted(0, 0, -1, -1), Qt.black)
            if index == self._current_item_index:
                painter.fillPath(circle_seed_item.path, Qt.green)
            else:
                painter.fillPath(circle_seed_item.path, Qt.red)