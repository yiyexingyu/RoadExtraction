# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 13:49
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : Exception.py
# @Project : RoadExtraction
# @Software: PyCharm


class NoInitializeError(Exception):

    def __init__(self):
        Exception.__init__(self)


class RadiusOverBorder(Exception):

    def __init__(self):
        Exception.__init__(self)
