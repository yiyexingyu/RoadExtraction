# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 20:05
# @Author  : 一叶星羽
# @Email   : 2958029539@qq.com
# @File    : GrayLevelCo-occurrenceMatrix.py
# @Project : RoadExtraction
# @Software: PyCharm

from enum import IntEnum
import numpy as np


class GLCMDirection(IntEnum):
    GLCM_HORIZATION = 0,  # 水平
    GLCM_VERTICAL = 1,  # 垂直
    GLCM_ANGLE45 = 2,  # 45 度角
    GLCM_ANGLE135 = 3  # 135 度角


class GrayLCM:
    """灰度共生矩阵的生成和其相关特征值的计算，如角阶距，对比度和熵等"""

    @staticmethod
    def _adjust_gray_level(src_gray_img: np.ndarray, gray_level: int):
        """对图片进行指定灰度级范围内进行归一化处理"""
        assert src_gray_img.ndim == 2

        gray_img = src_gray_img.astype(np.int32)
        if gray_img.max() >= gray_level:
            gray_img = (gray_img / gray_level).astype(np.int32)  # type: np.ndarray
        return gray_img

    @staticmethod
    def get_horizon_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成水平方向（0°）的共生矩阵 （偏移量默认是1）"""
        src_gray_img = GrayLCM._adjust_gray_level(src_gray_img, gray_level)
        shape = src_gray_img.shape

        result = np.zeros(shape, np.int32)
        for y in range(shape[0]):
            for x in range(shape[1] - 1):
                # 如果圆形图片，要进行格外判断
                row, clo = src_gray_img[y][x], src_gray_img[y][x+1]
                if not (0 <= row < shape[0]) or not (0 <= clo < shape[1]):
                    continue
                result[row][clo] += 1
        return result

    @staticmethod
    def get_vertical_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成垂直（90°）方向的共生矩阵 （偏移量默认是1）"""
        # shape = src_gray_img.shape
        src_gray_img = GrayLCM._adjust_gray_level(src_gray_img, gray_level)
        shape = src_gray_img.shape

        result = np.zeros(shape, np.int32)
        for y in range(shape[0] - 1):
            for x in range(shape[1]):
                # 如果圆形图片，要进行格外判断
                row, clo = src_gray_img[y][x], src_gray_img[y + 1][x]
                if not (0 <= row < shape[0]) or not (0 <= clo < shape[1]):
                    continue
                result[row][clo] += 1
        return result

    @staticmethod
    def get_45_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成45°方向的共生矩阵 （偏移量默认是1）"""
        # shape = src_gray_img.shape
        src_gray_img = GrayLCM._adjust_gray_level(src_gray_img, gray_level)
        shape = src_gray_img.shape

        result = np.zeros(shape, np.int32)
        for y in range(shape[0] - 1):
            for x in range(shape[1] - 1):
                # 如果圆形图片，要进行格外判断
                row, clo = src_gray_img[y][x], src_gray_img[y + 1][x + 1]
                if not (0 <= row < shape[0]) or not (0 <= clo < shape[1]):
                    continue
                result[row][clo] += 1
        return result

    @staticmethod
    def get_135_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成135°方向的共生矩阵 （偏移量默认是1）"""
        # shape = src_gray_img.shape
        src_gray_img = GrayLCM._adjust_gray_level(src_gray_img, gray_level)
        shape = src_gray_img.shape

        result = np.zeros(shape, np.int32)
        for y in range(shape[0] - 1):
            for x in range(1, shape[1]):
                # 如果圆形图片，要进行格外判断
                row, clo = src_gray_img[y][x], src_gray_img[y + 1][x - 1]
                if not (0 <= row < shape[0]) or not (0 <= clo < shape[1]):
                    continue
                result[row][clo] += 1
        return result

    @staticmethod
    def get_glcm(src_gray_img: np.ndarray, direction: GLCMDirection, gray_level: int) -> [np.ndarray, None]:
        """
        根据方向direction和灰度级gray_level生成源图片的灰度共生矩阵（偏移量默认是1）
        :param src_gray_img: 源图片矩阵
        :param direction: 在哪个方向上生成灰度共生矩阵
        :param gray_level: 共生矩阵的灰度级
        :return: 生成的灰度共生矩阵
        """
        if direction == GLCMDirection.GLCM_HORIZATION:
            return GrayLCM.get_horizon_glcm(src_gray_img, gray_level)
        if direction == GLCMDirection.GLCM_VERTICAL:
            return GrayLCM.get_vertical_glcm(src_gray_img, gray_level)
        if direction == GLCMDirection.GLCM_ANGLE45:
            return GrayLCM.get_45_glcm(src_gray_img, gray_level)
        if direction == GLCMDirection.GLCM_ANGLE135:
            return GrayLCM.get_135_glcm(src_gray_img, gray_level)
        return None

    @staticmethod
    def get_energy(gray_level_cm: np.ndarray):
        """计算灰度共生矩阵的角二阶距即能量"""
        # 这里可使用float32来降低内存的消耗
        # 归一化处理
        glcm_norm = gray_level_cm / np.sum(gray_level_cm)
        return np.sum(np.square(glcm_norm))

    @staticmethod
    def get_contrast(gray_level_cm: np.ndarray):
        """计算灰度共生矩阵的对比度"""
        glcm_norm = gray_level_cm / np.sum(gray_level_cm)
        shape = glcm_norm.sahpe
        result = 0.

        for y in range(shape[0]):
            for x in range(shape[1]):
                result += ((x - y) ** 2) * glcm_norm[x][y]
        return result

    @staticmethod
    def get_entropy(gray_level_cm: np.ndarray):
        """计算灰度共生矩阵的熵"""

        glcm_norm = gray_level_cm / np.sum(gray_level_cm)
        shape = glcm_norm.sahpe
        result = 0.

        for y in range(shape[0]):
            for x in range(shape[1]):
                if gray_level_cm[y][x] > 0:
                    result -= glcm_norm[y][x] * np.log2(glcm_norm[y][x])
        return result

    @staticmethod
    def get_road_texture_vector(src_road_img: np.ndarray):
        """
        计算道路的主要纹理特征，包括角二阶距、对比度和熵，并返回其纹理特征向量。
        在0、45、90、135°四个方向上求源图片矩阵的角二阶距、对比度和熵，然后分别
        求取它们的均值和标准差，共6个值共同组成了源图片的道路相关的纹理特征向量
        :param src_road_img: 源灰度图片矩阵,
        :return: 返回源图片的道路相关的纹理特征向量
        """
        assert src_road_img.ndim == 2

        # 首先求图片的灰度共生矩阵
        gray_level_cm_horizon = GrayLCM.get_horizon_glcm(src_road_img)
        gray_level_cm_vertical = GrayLCM.get_vertical_glcm(src_road_img)
        gray_level_cm_45 = GrayLCM.get_45_glcm(src_road_img)
        gray_level_cm_135 = GrayLCM.get_135_glcm(src_road_img)

        energy_horizon, contrast_horizon, entropy_horizon = GrayLCM.get_road_texture(gray_level_cm_horizon)
        energy_vertical, contrast_vertical, entropy_vertical = GrayLCM.get_road_texture(gray_level_cm_vertical)
        energy_45, contrast_45, entropy_45 = GrayLCM.get_road_texture(gray_level_cm_45)
        energy_135, contrast_135, entropy_135 = GrayLCM.get_road_texture(gray_level_cm_135)

        energy = [energy_horizon, energy_45, energy_vertical, energy_135]
        contrast = [contrast_horizon, contrast_45, contrast_vertical, contrast_135]
        entropy = [entropy_horizon, entropy_45, entropy_vertical, entropy_135]

        # 求均值
        energy_mean = np.mean(energy)
        contrast_mean = np.mean(contrast)
        entropy_mean = np.mean(entropy)

        # 求标准差
        energy_std = np.std(energy)
        contrast_std = np.std(contrast)
        entropy_std = np.std(entropy)

        return [energy_mean, contrast_mean, entropy_mean, energy_std, contrast_std, entropy_std]

    @staticmethod
    def get_road_texture(gray_level_cm: np.ndarray):
        """计算灰度共生矩阵的三个特征：角二阶距、对比度和熵"""
        # 归一化处理
        glcm_norm = gray_level_cm / np.sum(gray_level_cm)
        shape = glcm_norm.shape

        # 角二阶距
        energy = np.sum(np.square(glcm_norm))

        # 对比度和熵
        contrast, entropy = 0, 0
        for y in range(shape[0]):
            for x in range(shape[1]):
                contrast += ((x - y) ** 2) * glcm_norm[y][x]
                if gray_level_cm[y][x] > 0:
                    entropy -= glcm_norm[y][x] * np.log2(glcm_norm[y][x])
        return energy, contrast, entropy

    @staticmethod
    def glcm(arr, d_x, d_y, gray_level=16):
        """计算并返回归一化后的灰度共生矩阵"""
        max_gray = arr.max()
        height, width = arr.shape
        # 将uint8类型转换为float64，以免数据失真
        arr = arr.astype(np.float64)
        # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，
        # 减小灰度共生矩阵的大小。量化后灰度值范围：0 ~ gray_level - 1
        arr = arr * ( gray_level - 1) // max_gray
        ret = np.zeros([gray_level, gray_level])
        for j in range(height - abs(d_y)):
            for i in range(width - abs(d_x)):
                rows = arr[j][i].astype(int)
                cols = arr[j + d_y][i + d_x].astype(int)
                ret[rows][cols] += 1
        if d_x >= d_y:
            ret = ret / float(height * (width - 1))  # 归一化, 水平方向或垂直方向
        else:
            ret = ret / float((height - 1) * (width - 1))  # 归一化, 45度或135度方向
        return ret


def get_circle_spectral_vector(src_rgb_img: np.ndarray):
    """
    计算圆形道路种子的光谱特征，并返回其特征向量。
    分别计算r、g、b的均值和标准差，共6个值共同组成了源图片的道路相关的光谱特征向量
    :param src_rgb_img: 源图片，RGB格式
    """
    assert src_rgb_img.ndim >= 3
    red = src_rgb_img[:, :, 0]
    red = red[red >= 0]

    green = src_rgb_img[:, :, 1]
    green = green[green >= 0]

    blue = src_rgb_img[:, :, 2]
    blue = blue[blue >= 0]

    red_mean = np.mean(red)
    green_mean = np.mean(green)
    blue_mean = np.mean(blue)

    red_std = np.std(red)
    green_std = np.std(green)
    blue_std = np.std(blue)
    return [red_mean, green_mean, blue_mean, red_std, green_std, blue_std]


def get_spectral_vector(src_rgb_img: np.ndarray):
    """
    计算src_rgb_img的光谱特征，并返回其特征向量。
    分别计算r、g、b的均值和标准差，共6个值共同组成了源图片的道路相关的光谱特征向量
    :param src_rgb_img: 源图片，RGB格式
    """
    red = src_rgb_img[:, :, 0]
    green = src_rgb_img[:, :, 1]
    blue = src_rgb_img[:, :, 2]
    return [np.mean(red), np.mean(green), np.mean(blue), np.std(red), np.std(green), np.std(blue)]


if __name__ == '__main__':

    import random
    import time
    import os
    print(os.path.abspath("."))
    size = 1000
    num = 12
    test_array = np.zeros((size, size), np.uint8)

    for i in range(6):

        x_, y_ = (random.randint(0, size-1), random.randint(0, size-1))
        test_array[x_][y_] = num + i

        t = time.time()
        _pos = np.unravel_index(np.argmax(test_array), test_array.shape)
        max_num = test_array[_pos[0]][_pos[1]]

        dt = time.time() - t
        print("np get max num: ", max_num, "  time=", dt)

        _shape = test_array.shape
        max_num = 0
        t = time.time()
        for y_ in range(_shape[1]):
            for x_ in range(_shape[0]):
                num = test_array[y_][x_]
                if num > max_num:
                    max_num = num
        dt = time.time() - t
        print("my get max num: ", max_num, "  time=", dt)
        print("=========================================")
