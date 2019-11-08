# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 21:43
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : CircleSeedDetail.py
# @Project : RoadExtraction
# @Software: PyCharm


from PyQt5.QtWidgets import QLabel, QTextBrowser, QHBoxLayout, QDialog
from DetectObjects.CircleSeed import CircleSeed


class CircleSeedDetail(QDialog):

    def __init__(self, parent, circle_seed: CircleSeed):
        super(CircleSeedDetail, self).__init__(parent)
        self.setWindowTitle("圆形种子详细信息")
        self.setMinimumWidth(380)
        self.setMinimumHeight(480)
        self._circle_seed_info = QTextBrowser(self)

        layout = QHBoxLayout(self)
        layout.addWidget(self._circle_seed_info)
        self._init_text(circle_seed)
        self.setLayout(layout)

    def _init_text(self, circle_seed: CircleSeed):
        self._circle_seed_info.append(circle_seed.__str__())

        for index, child_seed in enumerate(circle_seed.child_seeds):
            self._circle_seed_info.append("子种子" + str(index+1) + ":\n" + child_seed.__str__())
