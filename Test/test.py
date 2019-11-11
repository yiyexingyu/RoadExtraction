# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 22:10
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : test.py
# @Project : RoadExtraction
# @Software: PyCharm

import sys
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage
from Core.RoadDetection import RoadDetectionEx


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

    sys.exit(app.exec_())


if __name__ == '__main__':
    test_main()
