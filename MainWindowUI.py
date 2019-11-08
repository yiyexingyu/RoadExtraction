# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 17:09
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : MainWindowUI.py
# @Project : RoadExtraction
# @Software: PyCharm

from PyQt5.QtWidgets import QMainWindow, QTextBrowser, QDockWidget, QHBoxLayout, QAction
from PyQt5.QtCore import Qt
from Test.ShowResultView import ShowResultView


class QMainWindowUI(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setWindowTitle("基于高分辨率的道路提取算法测试")
        self.setMinimumWidth(680)
        self.setMinimumHeight(480)

        self._init_window_content()
        self._init_menubar()

    def _init_window_content(self):
        self.show_result_view = ShowResultView(parent=self)
        self.setCentralWidget(self.show_result_view)

        self.show_result_dock = QDockWidget(self)
        self.h_box_layout = QHBoxLayout(self.show_result_dock)
        self.show_result_browser = QTextBrowser()
        self.h_box_layout.addWidget(self.show_result_browser)

        self.show_result_dock.setLayout(self.h_box_layout)
        self.addDockWidget(Qt.RightDockWidgetArea, self.show_result_dock)

    def _init_menubar(self):
        self._menubar = self.menuBar()

        self.open_image_action = QAction("图片")
        self.start_detect_action = QAction("开始检测")
        self.end_detect_action = QAction("终止检测")

        self._menubar.addAction(self.open_image_action)
        self._menubar.addAction(self.start_detect_action)
        self._menubar.addAction(self.end_detect_action)
