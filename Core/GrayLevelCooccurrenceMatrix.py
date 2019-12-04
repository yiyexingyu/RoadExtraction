# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 20:05
# @Author  : 一叶星羽
# @Email   : 2958029539@qq.com
# @File    : GrayLevelCo-occurrenceMatrix.py
# @Project : RoadExtraction
# @Software: PyCharm

import time
from numba import jit
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
    @jit
    def adjust_gray_level(src_gray_img: np.ndarray, gray_level: int):
        """对图片进行指定灰度级范围内进行归一化处理"""
        assert src_gray_img.ndim == 2

        gray_img = src_gray_img.astype(np.int32)
        while gray_img.max() >= gray_level:
            gray_img = (gray_img / gray_level).astype(np.int32)  # type: np.ndarray
        return gray_img

    @staticmethod
    @jit(forceobj=True)
    def get_vertical_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成垂直（90°）方向的共生矩阵 （偏移量默认是1）"""
        # shape = src_gray_img.shape
        src_gray_img = GrayLCM.adjust_gray_level(src_gray_img, gray_level)
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
    @jit(forceobj=True)
    def get_horizon_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成水平方向（0°）的共生矩阵 （偏移量默认是1）"""
        src_gray_img = GrayLCM.adjust_gray_level(src_gray_img, gray_level)
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
    @jit(forceobj=True)
    def get_45_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成45°方向的共生矩阵 （偏移量默认是1）"""
        # shape = src_gray_img.shape
        src_gray_img = GrayLCM.adjust_gray_level(src_gray_img, gray_level)
        shape = src_gray_img.shape

        result = np.zeros(shape, np.int32)
        for y in range(shape[0] - 1):
            for x in range(shape[1] - 1):
                # 如果圆形图片，要进行格外判断
                row, clo = src_gray_img[y][x], src_gray_img[y + 1][x + 1]
                if not (0 <= row < shape[0]) or not (0 <= clo < shape[1]):
                    continue
                result[row][clo] += 1
        np.printoptions()
        return result

    @staticmethod
    @jit(forceobj=True)
    def get_135_glcm(src_gray_img: np.ndarray, gray_level: int = 16) -> np.ndarray:
        """生成135°方向的共生矩阵 （偏移量默认是1）"""
        # shape = src_gray_img.shape
        src_gray_img = GrayLCM.adjust_gray_level(src_gray_img, gray_level)
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
    @jit
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
    @jit
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
    # @jit
    def get_entropy_of(gray_level_cm: np.ndarray):
        grey1 = gray_level_cm[:, :, 0, 0]
        grey1 = grey1[grey1 > 0]

        grey2 = gray_level_cm[:, :, 0, 1]
        grey2 = grey2[grey2 > 0]

        grey3 = gray_level_cm[:, :, 0, 2]
        grey3 = grey3[grey3 > 0]

        grey4 = gray_level_cm[:, :, 0, 3]
        grey4 = grey4[grey4 > 0]

        return [
            -np.sum(grey1 * np.log2(grey1)),
            -np.sum(grey2 * np.log2(grey2)),
            -np.sum(grey3 * np.log2(grey3)),
            -np.sum(grey4 * np.log2(grey4))
        ]

    @staticmethod
    @jit
    def get_entropy(gray_level_cm: np.ndarray):
        """计算灰度共生矩阵的熵"""

        glcm_norm = gray_level_cm / np.sum(gray_level_cm)
        shape = glcm_norm.sahpe
        result = 0.

        # 16 * 16 * 1 * 4

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
        src_road_img = GrayLCM.adjust_gray_level(src_road_img, gray_level=16)
        gray_level_cm_horizon = GrayLCM.get_horizon_glcm(src_road_img)
        gray_level_cm_vertical = GrayLCM.get_vertical_glcm(src_road_img)
        gray_level_cm_45 = GrayLCM.get_45_glcm(src_road_img)
        gray_level_cm_135 = GrayLCM.get_135_glcm(src_road_img)

        return GrayLCM.calculate_road_texture_vector(
            [gray_level_cm_horizon, gray_level_cm_vertical, gray_level_cm_45, gray_level_cm_135])

    @staticmethod
    @jit(forceobj=True)
    def calculate_road_texture_vector(gray_level_cms: list):
        gray_level_cm_horizon = gray_level_cms[0]
        gray_level_cm_vertical = gray_level_cms[1]
        gray_level_cm_45 = gray_level_cms[2]
        gray_level_cm_135 = gray_level_cms[3]

        # st = time.time()
        energy_horizon, contrast_horizon, entropy_horizon = GrayLCM.get_road_texture(gray_level_cm_horizon)
        # print("计算纹理特征：", time.time() - st)

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
    @jit
    def get_road_texture(gray_level_cm: np.ndarray, normal=True):
        """计算灰度共生矩阵的三个特征：角二阶距、对比度和熵"""
        # 归一化处理
        if normal:
            glcm_norm = gray_level_cm / np.sum(gray_level_cm)
        else:
            glcm_norm = gray_level_cm
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
    @jit(forceobj=True)
    def greycoprops(P, prop='contrast'):
        # check_nD(P, 4, 'P')

        (num_level, num_level2, num_dist, num_angle) = P.shape
        if num_level != num_level2:
            raise ValueError('num_level and num_level2 must be equal.')
        if num_dist <= 0:
            raise ValueError('num_dist must be positive.')
        if num_angle <= 0:
            raise ValueError('num_angle must be positive.')

        # normalize each GLCM
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

        # create weights for specified property
        I, J = np.ogrid[0:num_level, 0:num_level]
        if prop == 'contrast':
            weights = (I - J) ** 2
        elif prop == 'dissimilarity':
            weights = np.abs(I - J)
        elif prop == 'homogeneity':
            weights = 1. / (1. + (I - J) ** 2)
        elif prop in ['ASM', 'energy', 'correlation']:
            pass
        else:
            raise ValueError('%s is an invalid property' % (prop))

        # compute property for each GLCM
        if prop == 'energy':
            asm = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
            results = np.sqrt(asm)
        elif prop == 'ASM':
            results = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
        elif prop == 'correlation':
            results = np.zeros((num_dist, num_angle), dtype=np.float64)
            I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
            J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
            diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
            diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

            std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                               axes=(0, 1))[0, 0])
            std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                               axes=(0, 1))[0, 0])
            cov = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                     axes=(0, 1))[0, 0]

            # handle the special case of standard deviations near zero
            mask_0 = std_i < 1e-15
            mask_0[std_j < 1e-15] = True
            results[mask_0] = 1

            # handle the standard case
            mask_1 = mask_0 == False
            results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
        elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
            weights = weights.reshape((num_level, num_level, 1, 1))
            results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

        return results

    @staticmethod
    @jit
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


@jit(forceobj=True)
def adjust_gray_level(src_gray_img: np.ndarray, gray_level: int):
    """对图片进行指定灰度级范围内进行归一化处理"""
    assert src_gray_img.ndim == 2

    gray_img = src_gray_img.astype(np.int32)
    while gray_img.max() >= gray_level:
        gray_img = (gray_img / gray_level)
    gray_img = gray_img.astype(np.int32)
    return gray_img


@jit(forceobj=True)
def grey_comatrix(src_gray_img, level=16):

    result = np.zeros((4, level, level))
    height, width = src_gray_img.shape

    # 0°
    for y in range(height):
        for x in range(width - 1):
            # 如果圆形图片，要进行格外判断
            row, clo = src_gray_img[y][x], src_gray_img[y][x + 1]
            if row < 0 or clo < 0:
                continue
            result[0][row][clo] += 1

    # 45°
    for y in range(height - 1):
        for x in range(width - 1):
            # 如果圆形图片，要进行格外判断
            row, clo = src_gray_img[y][x], src_gray_img[y + 1][x + 1]
            if row < 0 or clo < 0:
                continue
            result[1][row][clo] += 1

    # 90°
    for y in range(height - 1):
        for x in range(width):
            # 如果圆形图片，要进行格外判断
            row, clo = src_gray_img[y][x], src_gray_img[y + 1][x]
            if row < 0 or clo < 0:
                continue
            result[2][row][clo] += 1

    # 135°
    for y in range(height - 1):
        for x in range(width - 1):
            # 如果圆形图片，要进行格外判断
            row, clo = src_gray_img[y][x], src_gray_img[y + 1][x + 1]
            if row < 0 or clo < 0:
                continue
            result[3][row][clo] += 1
    return result


@jit(forceobj=True)
def get_road_texture(gray_level_cm: np.ndarray):
    """计算灰度共生矩阵的三个特征：角二阶距、对比度和熵"""
    glcm_norm = gray_level_cm
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


def road_texture_vector(src_grey_img, level=16):
    src_grey_img = adjust_gray_level(src_grey_img, level)

    co_matrix = grey_comatrix(src_grey_img, level)
    feature1 = get_road_texture(co_matrix[0])
    feature2 = get_road_texture(co_matrix[1])
    feature3 = get_road_texture(co_matrix[2])
    feature4 = get_road_texture(co_matrix[3])
    feature = np.vstack((feature1, feature2, feature3, feature4))
    energy, contrast, entropy = feature[:, 0], feature[:, 1], feature[:, 2]

    energy_mean = np.add.reduce(energy) / 4
    contrast_mean = np.add.reduce(contrast) / 4
    entropy_mean = np.add.reduce(entropy) / 4

    energy_std = np.std(energy)
    contrast_std = np.std(contrast)
    entropy_std = np.std(entropy)
    return [energy_mean, contrast_mean, entropy_mean, energy_std, contrast_std, entropy_std]


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
