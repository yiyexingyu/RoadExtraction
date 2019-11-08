# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 17:36
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : RoadExtraction.py
# @Project : RoadExtraction
# @Software: PyCharm

import os
from PyQt5.QtWidgets import QGraphicsScene, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QRectF
from MainWindowUI import QMainWindowUI
from Test.ShowResultLabel import ShowResultItem
from Test.test import RoadDetectThread
from Core.RoadDetection import RoadDetection


class RoadExtraction(QMainWindowUI):

    def __init__(self):
        super(RoadExtraction, self).__init__()

        self.img_last_dir = "./"
        self.scene = QGraphicsScene(self)
        self.show_result_view.setScene(self.scene)

        self.show_result_item = ShowResultItem()
        self.scene.addItem(self.show_result_item)
        self.show_result_view.main_item = self.show_result_item

        self._road_detection = RoadDetection()
        self._road_detection.circle_seeds_generated.connect(self.show_result_item.new_seed_generated)

        self._road_detect_thread = RoadDetectThread(self._road_detection)
        self._road_detect_thread.road_detect_finished_signal.connect(self.show_result_item.road_detection_finished)
        self._road_detect_thread.road_detect_finished_signal.connect(self.show_finished_info)
        self.show_result_item.init_circle_seed_signal.connect(self.start_detect_action.setEnabled)

        self.show_result_browser.append("请输入初始化种子")
        self.change_image("D:/4.png")
        self.connect_modify()

    def connect_modify(self):
        self.open_image_action.triggered.connect(self.open_image)
        self.start_detect_action.triggered.connect(self.start_road_detect)
        # self.end_detect_action.triggered.connect(self.end_detect)

    def open_image(self):
        file_format = "Image files (*.png *.jpg *tif)"
        image_file = QFileDialog.getOpenFileName(
            self, "选择原始图片",
            self.img_last_dir,
            file_format
        )[0]

        self.change_image(image_file)

    def show_finished_info(self):
        self._road_detection.set_init(False)
        self.show_result_item.is_road_detecting = False
        QMessageBox.information(self, "道路检测", "道路检测完成！")

    def start_road_detect(self):
        self.start_detect_action.setEnabled(False)
        init_circle_seed = self.show_result_item.start_road_detect()

        print(init_circle_seed)
        if init_circle_seed:
            print(init_circle_seed)
            self._road_detection.initialize(init_circle_seed.position, init_circle_seed.radius)
            self._road_detect_thread.start()

    def change_image(self, image_file):

        if image_file and os.path.exists(image_file):
            self.img_last_dir = os.path.dirname(image_file)
            image = QImage(image_file)
            self.scene.setSceneRect(QRectF(image.rect()))
            self.show_result_item.image = image
            self._road_detection.image = image


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    win = RoadExtraction()
    win.show()
    sys.exit(app.exec_())
