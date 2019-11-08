# -*- coding: utf-8 -*-
# @Time    : 2019/11/3 19:22
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : MorphologicalFiltering.py
# @Project : RoadExtraction
# @Software: PyCharm

import cv2
from PyQt5.QtGui import QImage
from DetectObjects.CircleSeed import CircleSeed
from DetectObjects.Utils import qimage2cvmat


def morphological_open_option(src_img: QImage, circle_seed: CircleSeed):
    """形态学处理： 开运算"""
    cv_img = qimage2cvmat(src_img)


