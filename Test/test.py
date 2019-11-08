# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 22:10
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : test.py
# @Project : RoadExtraction
# @Software: PyCharm

import sys
import threading
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QTextBrowser
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from Core.RoadDetection import RoadDetection, CircleSeed, RoadDetectionEx
from Test.ShowResultLabel import ShowResultLabel


class ShowImageWidget(QWidget):

    new_seeds_generated = pyqtSignal(CircleSeed)

    def __init__(self, image: QImage):
        super(ShowImageWidget, self).__init__()
        self.setWindowTitle("道路提取算法测试")
        self.setFixedWidth(image.width() + 320)
        self.setFixedHeight(image.height())
        self._horizon_layout = QHBoxLayout(self)
        self._horizon_layout.setContentsMargins(0, 0, 5, 5)
        self.setLayout(self._horizon_layout)

        self._image = image
        self._image_label = ShowResultLabel(image, self)
        self._image_label.setScaledContents(True)
        self._image_label.setGeometry(0, 0, image.width(), image.height())
        self.new_seeds_generated.connect(self._image_label.new_seed_generated)
        self._image_label.setPixmap(QPixmap.fromImage(image))
        self._horizon_layout.addWidget(self._image_label)

        self._show_info_browser = QTextBrowser(self)
        self._horizon_layout.addWidget(self._show_info_browser)

        self._road_detection = RoadDetectionEx(image)
        self._road_detection.circle_seeds_generated.connect(self.show_generated_seeds_info)
        self._road_detection.circle_seeds_generated.connect(self._image_label.new_seed_generated)

        self._road_detect_thread = RoadDetectThread(self._road_detection)
        self._road_detect_thread.road_detect_finished_signal.connect(self._image_label.road_detection_finished)
        self._road_detect_thread.road_detect_finished_signal.connect(self.road_detect_finished)

        self._image_label.start_road_detection_signal.connect(self.about_to_road_detect)
        self._image_label.circle_seed_clicked_signal.connect(self.show_selected_seed_info)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Q and event.modifiers() & Qt.ControlModifier:
            if self._road_detect_thread.isRunning():
                self._road_detect_thread.pause()
                event.accept()
        elif event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self._road_detect_thread.resume()
            event.accept()
        else:
            self._image_label.keyPressEvent(event)

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        self._image_label.keyReleaseEvent(a0)

    def about_to_road_detect(self, init_circle_seed: CircleSeed):
        init_circle_seed.init_circle_seed(self._image)
        self._road_detection.initialize(init_circle_seed.position, init_circle_seed.radius)
        self._road_detect_thread.start()
        self._show_info_browser.append("==============================")
        self._show_info_browser.append("开始进行道路检测....")

    def add_circle_seed(self, circle_seed: CircleSeed):
        self._image_label.add_circle_seed(circle_seed)

    def show_selected_seed_info(self, index, circle_seed):
        self._show_info_browser.append(str(index) + "号子种子信息：\n" + circle_seed.__str__())

    def show_generated_seeds_info(self, generated_seed: CircleSeed):
        self._show_info_browser.append("生成子种子：\n" + generated_seed.__str__())

    def road_detect_finished(self, road_detection: RoadDetection):
        self._show_info_browser.append(
            "道路检测结束： \n" + "共产生圆形种子：" + str(len(road_detection.get_seed_list()) - 1) + " 个")
        self._show_info_browser.append("==============================")


class RoadDetectThread(QThread):

    road_detect_finished_signal = pyqtSignal(RoadDetectionEx)

    def __init__(self, road_detection: RoadDetectionEx):
        super(RoadDetectThread, self).__init__()
        self._road_detection = road_detection
        self.__flag = threading.Event()  # 用于暂停线程的标识
        self.__flag.set()  # 设置为True
        self.__running = threading.Event()  # 用于停止线程的标识
        self.__running.set()  # 将running设置为True

    def run(self) -> None:
            print("开始进行道路检测....")
            while self.__running.isSet():
                self.__flag.wait()  # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
                if not self._road_detection.road_detection_one_step():
                    break
            print("线程任务完成！")
            self.road_detect_finished_signal.emit(self._road_detection)

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞
        print("pause")

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞
        print("resume")

    def stop(self):
        # self.__flag.set()  # 将线程从暂停状态恢复, 如果已经暂停的话（要是停止的话我就直接让他停止了，
        # 干嘛还要执行这一句语句啊，把这句注释了之后就没有滞后现象了。）
        self.__running.clear()  # 设置为False


def test_main():
    # 到目前结果最好的位置和半径：[(799, 635), 11] [(605, 426), 11]
    image_path = "F:/RoadDetectionTestImg/4.png"
    image1 = QImage(image_path)
    # image1 = image1.convertToFormat(QImage.Format_Grayscale8)

    app = QApplication(sys.argv)
    window = ShowImageWidget(image1)

    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    from setuptools import find_packages
    for pk in find_packages():
        print(pk)
    test_main()
