# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 13:49
# @Author  : 何盛信
# @Email   : 2958029539@qq.com
# @File    : Exception.py
# @Project : RoadExtraction
# @Software: PyCharm


class NoInitializeError(RuntimeError):

    """ 进行道路跟踪检测时，没有进行初始化 """
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        pass


class RadiusOverBorder(Exception):

    def __init__(self):
        Exception.__init__(self)
