# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 17:18
# @Author  : 一叶星羽
# @Email   : 2958029539@qq.com
# @File    : BezierCurveTest.py
# @Project : RoadExtraction
# @Software: PyCharm

import typing
from math import sqrt, nan, isnan
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPainterPath, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QPoint, QPointF


def point_to_point_distance(point1: [QPoint, QPointF], point2: [QPoint, QPointF]) -> float:
    return sqrt((point1.x() - point2.x()) ** 2 + (point1.y() - point2.y()) ** 2)


def slope_of_two_point(point1: typing.Union[QPointF, QPoint], point2: typing.Union[QPointF, QPoint]):
    try:
        return (point1.y() - point2.y()) / (point1.x() - point2.x())
    except ZeroDivisionError:
        return nan


class LineStatue:

    def __init__(self):
        self._is_draw = True
        self._is_selected_line = False
        self._is_selected_start_pos = False
        self._is_selected_end_pos = False

    @property
    def is_mouse_enter(self):
        return self._is_selected_line or self._is_selected_start_pos or self._is_selected_end_pos

    @property
    def is_draw(self) -> bool:
        return self._is_draw

    @is_draw.setter
    def is_draw(self, draw: bool):
        self._is_draw = draw

    @property
    def selected_line(self):
        return self._is_selected_line

    @selected_line.setter
    def selected_line(self, is_selected_line):
        self._is_selected_line = is_selected_line
        if self._is_selected_line:
            self._is_selected_start_pos = False
            self._is_selected_end_pos = False

    @property
    def selected_start_pos(self):
        return self._is_selected_start_pos

    @selected_start_pos.setter
    def selected_start_pos(self, is_selected_start_pos):
        self._is_selected_start_pos = is_selected_start_pos
        if self._is_selected_start_pos:
            self._is_selected_end_pos = False
            self._is_selected_line = False

    @property
    def selected_end_pos(self):
        return self._is_selected_end_pos

    @selected_end_pos.setter
    def selected_end_pos(self, is_selected_end_pos):
        self._is_selected_end_pos = is_selected_end_pos
        if self._is_selected_end_pos:
            self._is_selected_start_pos = False
            self._is_selected_line = False


class RoadStatue:

    def __init__(self):
        self._is_draw = True
        self._is_fill = True
        self._is_mouse_enter = False
        self._is_selected = False

    @property
    def is_draw(self):
        return self._is_draw

    @is_draw.setter
    def is_draw(self, draw):
        self._is_draw = draw

    @property
    def is_fill(self):
        return self._is_fill

    @is_fill.setter
    def is_fill(self, fill):
        self._is_fill = fill

    @property
    def is_mouse_enter(self):
        return self._is_mouse_enter

    @is_mouse_enter.setter
    def is_mouse_enter(self, mouse_enter):
        self._is_mouse_enter = mouse_enter
        if not self._is_mouse_enter:
            self._is_selected = False

    @property
    def is_selected(self):
        return self._is_selected

    @is_selected.setter
    def is_selected(self, selected):
        self._is_selected = selected


class ControlLine(QObject):
    tip_radius = 5
    start_pos_change_signal = pyqtSignal(QPointF)
    end_pos_change_signal = pyqtSignal(QPointF)

    def __init__(self, start_point: typing.Union[QPointF, QPoint], end_point: typing.Union[QPointF, QPoint]):
        super(ControlLine, self).__init__()

        # 控制线的主要信息，包括起始点、线条状态信息、线条的控制点
        self._start_point = start_point
        self._end_point = end_point
        self._line_statue = LineStatue()
        self._control_poses = [start_point, end_point]

        # 控制线的额外信息
        self._mouse_pressed_flag = False
        self._mouse_offset = QPoint()

    def mouse_press_event(self, position: QPoint):
        if self.line_statue.is_mouse_enter:
            self._mouse_pressed_flag = True
            self._mouse_offset = position
        else:
            self._mouse_pressed_flag = False

    def mouse_move_event(self, position: QPoint):
        if self._mouse_pressed_flag:
            if self.line_statue.selected_start_pos:
                self.start_pos = self._mouse_offset
            elif self.line_statue.selected_end_pos:
                self.end_pos = self._mouse_offset
            elif self.line_statue.selected_line:
                mouse_offset = position - self._mouse_offset
                self.offset(mouse_offset)
            self._mouse_offset = position
        else:
            distance = self.distance(position)
            if distance <= 5:
                distance1 = point_to_point_distance(self.start_pos, position)
                distance2 = point_to_point_distance(self.end_pos, position)
                if distance1 < distance2 and distance1 <= 6:
                    self.line_statue.selected_start_pos = True
                elif distance2 < distance1 and distance2 <= 6:
                    self.line_statue.selected_end_pos = True
                else:
                    self.line_statue.selected_line = True
            else:
                self.line_statue.selected_start_pos = False
                self.line_statue.selected_end_pos = False
                self.line_statue.selected_line = False

    def mouse_release_event(self, position: QPoint):
        self._mouse_pressed_flag = False

    def paint(self, painter: QPainter):
        painter.save()
        pen = QPen(Qt.red)
        pen.setWidth(1)
        if self.line_statue.selected_start_pos:
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawEllipse(self.start_pos, ControlLine.tip_radius, ControlLine.tip_radius)
        elif self.line_statue.selected_end_pos:
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawEllipse(self.end_pos, ControlLine.tip_radius, ControlLine.tip_radius)
        elif self.line_statue.selected_line:
            pen.setWidth(4)
            painter.setPen(pen)
        painter.setPen(pen)
        painter.drawLine(self.start_pos, self.end_pos)
        painter.restore()

    def contains(self, point: [QPointF, QPoint]) -> bool:
        """"""

    def distance(self, point: QPointF) -> float:
        """求点到直线的距离"""
        foot = self.perpendicular(point)
        return sqrt((point.x() - foot.x()) ** 2 + (point.y() - foot.y()) ** 2)

    def perpendicular(self, point: [QPointF, QPoint]) -> [QPointF, QPoint]:
        """计算给定的点和本线段的垂足"""
        x0 = point.x()
        y0 = point.y()

        x1 = self._start_point.x()
        y1 = self._start_point.y()

        x2 = self._end_point.x()
        y2 = self._end_point.y()

        if x1 == x2 and y1 == y2:
            return 0
        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        x = k * (x2 - x1) + x1
        y = k * (y2 - y1) + y1
        return QPointF(x, y)

    def offset(self, offset: QPointF):
        self.start_pos = self._start_point + offset
        self.end_pos = self._end_point + offset

    @property
    def start_pos(self) -> QPoint:
        return self._start_point

    @start_pos.setter
    def start_pos(self, new_start_pos: QPointF):
        if self._start_point == new_start_pos:
            return
        self._start_point = new_start_pos
        self.start_pos_change_signal.emit(QPointF(new_start_pos))

    @property
    def end_pos(self) -> QPointF:
        return self._end_point

    @end_pos.setter
    def end_pos(self, new_end_pos):
        if self._end_point == new_end_pos:
            return
        self._end_point = new_end_pos
        self.end_pos_change_signal.emit(QPointF(new_end_pos))

    @property
    def line_statue(self) -> LineStatue:
        return self._line_statue


class RoadPainterPath(QPainterPath):
    interval = 12
    mouse_enter_color = QColor(110, 11, 208, 60)
    general_color = QColor(0, 0, 232, 60)
    selected_color = QColor(243, 189, 18, 60)

    def __init__(self, control_line1: ControlLine = None, control_line2: ControlLine = None, center_line=None):
        super(RoadPainterPath, self).__init__()
        self._control_line1 = control_line1
        self._control_line2 = control_line2
        self._center_line = center_line
        self._road_statue = RoadStatue()
        self._init_road_path()
        self._offset = QPointF()

        self._control_line1.start_pos_change_signal.connect(self.reset)
        self._control_line1.end_pos_change_signal.connect(self.reset)
        self._control_line2.start_pos_change_signal.connect(self.reset)
        self._control_line2.end_pos_change_signal.connect(self.reset)

    def _init_road_path(self):
        self.moveTo(self.control_line1.start_pos)
        self.lineTo(self._control_line1.end_pos)
        self.lineTo(self._control_line2.end_pos)
        self.lineTo(self._control_line2.start_pos)

    def reset(self):
        self.clear()
        self._init_road_path()

    def mouse_press_event(self, position: QPoint):
        self._control_line1.mouse_press_event(position)
        self._control_line2.mouse_press_event(position)
        if not self.control_line1.line_statue.is_mouse_enter and not self._control_line2.line_statue.is_mouse_enter:
            self._road_statue.is_selected = self._road_statue.is_mouse_enter
            if self._road_statue.is_selected:
                self._offset = position

    def mouse_move_event(self, position: QPoint):
        self._control_line1.mouse_move_event(position)
        self._control_line2.mouse_move_event(position)
        if not self.control_line1.line_statue.is_mouse_enter and not self._control_line2.line_statue.is_mouse_enter:
            self._road_statue.is_mouse_enter = self.contains(position)
        else:
            self._road_statue.is_mouse_enter = False
        if self._road_statue.is_selected:
            mouse_offset = position - self._offset
            self._control_line1.offset(mouse_offset)
            self._control_line2.offset(mouse_offset)
            self._offset = position

    def mouse_release_event(self, position: QPoint):
        self._control_line1.mouse_release_event(position)
        self._control_line2.mouse_release_event(position)
        self._road_statue.is_selected = False
        self._road_statue.is_mouse_enter = self.contains(position)

    def paint(self, painter: QPainter):
        painter.save()
        self._control_line1.paint(painter)
        self._control_line2.paint(painter)

        color = RoadPainterPath.general_color
        if self._road_statue.is_selected:
            color = RoadPainterPath.selected_color
        elif self._road_statue.is_mouse_enter:
            color = RoadPainterPath.mouse_enter_color

        if self._road_statue.is_fill:
            painter.fillPath(self, color)
        else:
            pen = QPen(color)
            painter.setPen(pen)
            painter.drawPath(self)
        painter.restore()

    @staticmethod
    def create(point1: typing.Union[QPointF, QPoint], point2: typing.Union[QPointF, QPoint]):
        """
        :param point1:
        :param point2:
        :return:
        :rtype: RoadPainterPath
        """
        x1, y1 = point1.x(), point1.y()
        x2, y2 = point2.x(), point2.y()
        assert x1 != x2 and y1 != y2

        # if x1 == x2:
        #     # 如果斜率为0， 则说明直线平行于x轴
        #     control_point1 = QPointF(point1.x(), point1.y() + RoadPainterPath.interval)
        #     control_point2 = QPointF(point2.x(), point2.y() + RoadPainterPath.interval)
        #     control_point3 = QPointF(point1.x(), point1.y() - RoadPainterPath.interval)
        #     control_point4 = QPointF(point2.x(), point2.y() - RoadPainterPath.interval)
        # elif y1 == y2:
        #     # 如果斜率为无穷大，则说明直线平行于y轴
        #     control_point1 = QPointF(point1.x() - RoadPainterPath.interval, point1.y())
        #     control_point2 = QPointF(point2.x() - RoadPainterPath.interval, point2.y())
        #     control_point3 = QPointF(point1.x() + RoadPainterPath.interval, point1.y())
        #     control_point4 = QPointF(point2.x() + RoadPainterPath.interval, point2.y())
        # else:
        distance = point_to_point_distance(point1, point2)
        offset_x = RoadPainterPath.interval * ((y2 - y1) / distance)
        offset_y = RoadPainterPath.interval * ((x2 - x1) / distance)

        offset1 = QPointF(offset_x, -offset_y)
        offset2 = QPointF(-offset_x, offset_y)

        control_point1 = point1 + offset2
        control_point2 = point2 + offset2
        control_point3 = point1 + offset1
        control_point4 = point2 + offset1

        line1 = ControlLine(control_point1, control_point2)
        line2 = ControlLine(control_point3, control_point4)
        road_path = RoadPainterPath(line1, line2)
        return road_path

    @property
    def control_line1(self) -> ControlLine:
        return self._control_line1

    @control_line1.setter
    def control_line1(self, new_line: ControlLine):
        self._control_line1 = new_line
        self._control_line1.start_pos_change_signal.connect(self.reset)
        self._control_line1.end_pos_change_signal.connect(self.reset)

    @property
    def control_line2(self) -> ControlLine:
        return self._control_line2

    @control_line2.setter
    def control_line2(self, new_line: ControlLine):
        self._control_line2 = new_line
        self._control_line2.start_pos_change_signal.connect(self.reset)
        self._control_line2.end_pos_change_signal.connect(self.reset)

    @property
    def center_line(self):
        return self._center_line

    @center_line.setter
    def center_line(self, new_center_line):
        self._center_line = new_center_line

    @property
    def road_statue(self):
        return self._road_statue


class BezierCurveLabel(QLabel):

    def __init__(self):
        super(BezierCurveLabel, self).__init__()
        self.setMouseTracking(True)
        self._init_line = []
        self._bezier_path = None

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        self._init_line = []
        self._bezier_path = None
        self.update()
        ev.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()
        pos = event.pos()

        if self._bezier_path is None:
            self._init_line = [pos, pos]
        else:
            self._bezier_path.mouse_press_event(pos)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()
        pos = event.pos()
        if self._bezier_path is None and self._init_line:
            self._init_line[1] = pos
        elif self._bezier_path is not None:
            self._bezier_path.mouse_move_event(pos)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        event.accept()
        if self._bezier_path is None:
            try:
                self._bezier_path = RoadPainterPath.create(self._init_line[0], self._init_line[1])
            except AssertionError:
                self._bezier_path = None
        else:
            self._bezier_path.mouse_release_event(event.pos())

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)
        if self._bezier_path is None and self._init_line:
            painter.drawLine(self._init_line[0], self._init_line[1])
        if self._bezier_path is not None:
            self._bezier_path.paint(painter)

        p = QPainterPath()
        p.arcMoveTo(100, 100, 100, 50, 30)
        p.arcTo(100, 100, 100, 50, 30, 120)

        n = QPainterPath()
        n.moveTo(100, 110)
        n.lineTo(200, 110)

        painter.drawPath(p)
        painter.drawPath(n)

        k = p & n
        pen.setWidth(4)
        pen.setColor(Qt.blue)
        painter.setPen(pen)
        painter.drawPath(k)
        print(k.isEmpty())


if __name__ == '__main__':
    import sys

    app = QApplication([])
    win = BezierCurveLabel()
    win.setWindowTitle("贝赛尔曲线测试")
    win.setGeometry(360, 120, 680, 480)
    win.show()

    sys.exit(app.exec_())
