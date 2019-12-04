# -*- encoding: utf-8 -*-
# @File    : FeatureExtraction.py
# @Time    : 2019/11/20 15:48
# @Author  : 一叶星羽
# @Email   : h0670131005@gmail.com
# @Software: PyCharm

import time
import cv2
import numpy as np
from numba import jit
from skimage.feature import greycomatrix
from Core.GrayLevelCooccurrenceMatrix import GrayLCM, get_spectral_vector


def feature_extraction2(image: np.ndarray, level:int = 16, kernel: int = 5):
    """
        遥感图像特征向量提取，包括rgb的均值和标准差、灰度共生矩阵的角二阶矩、对比度和熵
        使用一个8*8的矩阵对图片的每个像素生成灰度共生矩阵、进行特征提取
        :param image: 原图片, ndarray格式，rgb三通道
        :type image: np.ndarray
        :param level: 灰度共生矩阵的级别，默认是16
        :param kernel: 移动窗口的大小，默认是5x5，kernel应该为奇数
        :return: 返回每个像素的特征向量，shape=[image.shape[0] * image.shape[1], 6, 6]
        :rtype: np.ndarray
    """
    result = []
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    padding_num = kernel // 2
    height, width = image.shape[0], image.shape[1]

    rgb_img = np.ones((image.shape[0] + padding_num * 2, image.shape[1] + padding_num * 2, 3))
    rgb_img[..., 0] = np.pad(image[..., 0], padding_num, "mean")
    rgb_img[..., 1] = np.pad(image[..., 1], padding_num, "mean")
    rgb_img[..., 2] = np.pad(image[..., 2], padding_num, "mean")
    rgb_img = rgb_img.astype(np.uint8)
    grey_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    grey_img = GrayLCM.adjust_gray_level(grey_img, gray_level=level)

    rgb_img = np.ascontiguousarray(rgb_img)
    grey_img = np.ascontiguousarray(grey_img)

    rgb_pixels_slice = np.ones((height * width, kernel, kernel, 3))
    grey_pixels_slice = np.ones((height * width, kernel, kernel))

    st = time.time()
    for y in range(0, image.shape[0] - kernel):
        y_stop = y + kernel
        for x in range(0, image.shape[1] - kernel):
            x_stop = x + kernel
            index = y * width + x
            rgb_pixels_slice[index] = rgb_img[y: y_stop, x: x_stop, :]
            grey_pixels_slice[index] = grey_img[y: y_stop, x: x_stop]
    print("切片时间：", time.time() - st)


@jit(forceobj=True)
def feature_extraction(image: np.ndarray, level=16, kernel: int = 5) -> np.ndarray:
    """
    遥感图像特征向量提取，包括rgb的均值和标准差、灰度共生矩阵的角二阶矩、对比度和熵
    使用一个8*8的矩阵对图片的每个像素生成灰度共生矩阵、进行特征提取
    :param image: 原图片, ndarray格式，rgb三通道
    :type image: np.ndarray
    :param level: 灰度共生矩阵的级别，默认是16
    :param kernel: 移动窗口的大小，默认是5x5，kernel应该为奇数
    :return:
    :rtype: np.ndarray
    """
    # result = []
    # angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    padding_num = kernel // 2

    rgb_img = np.ones((image.shape[0] + padding_num * 2, image.shape[1] + padding_num * 2, 3))
    # 用 均值 填充四周
    rgb_img[..., 0] = np.pad(image[..., 0], padding_num, "mean")
    rgb_img[..., 1] = np.pad(image[..., 1], padding_num, "mean")
    rgb_img[..., 2] = np.pad(image[..., 2], padding_num, "mean")
    rgb_img = rgb_img.astype(np.uint8)
    grey_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    grey_img = GrayLCM.adjust_gray_level(grey_img, gray_level=level)

    rgb_img = np.ascontiguousarray(rgb_img)
    grey_img = np.ascontiguousarray(grey_img)

    result = _extraction_loop(grey_img, rgb_img, level, padding_num)
    # print(result.shape)
    # print(result[0])
    # for y in range(padding_num, grey_img.shape[0] - padding_num):
    #     for x in range(padding_num, grey_img.shape[1] - padding_num):
    #         grey_pixel_kernel = grey_img[y - padding_num: y + padding_num + 1, x - padding_num: x + padding_num + 1]
    #         rgb_pixel_kernel = rgb_img[y - padding_num: y + padding_num + 1, x - padding_num: x + padding_num + 1, :]
    #
    #         # 计算纹理特征
    #         grey_co_matrices = greycomatrix(grey_pixel_kernel, [1], angles, level, normed=True)
    #         road_vector = GrayLCM.calculate_road_texture_vector(
    #             [grey_co_matrices[:, :, 0, 0], grey_co_matrices[:, :, 0, 1],
    #              grey_co_matrices[:, :, 0, 2], grey_co_matrices[:, :, 0, 3]]
    #         )
    #
    #         # 计算光谱特征
    #         spectral_vector = get_spectral_vector(rgb_pixel_kernel)
    #         road_vector.extend(spectral_vector)
    #         result.append(road_vector)
    return np.array(result)


@jit(forceobj=True)
def _extraction_loop(grey_img, rgb_img, level, padding_num):
    """
    :param grey_img:
    :param rgb_img:
    :param level:
    :param padding_num:
    :return:
    """
    result = []
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for y in range(padding_num, grey_img.shape[0] - padding_num):
        for x in range(padding_num, grey_img.shape[1] - padding_num):
            grey_pixel_kernel = grey_img[y - padding_num: y + padding_num + 1, x - padding_num: x + padding_num + 1]
            rgb_pixel_kernel = rgb_img[y - padding_num: y + padding_num + 1, x - padding_num: x + padding_num + 1, :]

            # 计算纹理特征
            grey_co_matrices = greycomatrix(grey_pixel_kernel, [1], angles, level, normed=True)

            # road_vector = calculate_texture_vector(grey_co_matrices)
            road_vector = GrayLCM.calculate_road_texture_vector(
                [grey_co_matrices[:, :, 0, 0], grey_co_matrices[:, :, 0, 1],
                 grey_co_matrices[:, :, 0, 2], grey_co_matrices[:, :, 0, 3]]
            )

            # 计算光谱特征
            spectral_vector = get_spectral_vector(rgb_pixel_kernel)
            road_vector.extend(spectral_vector)
            result.append(road_vector)

    return result


def calculate_texture_vector(grey_co_matrices):
    energy = GrayLCM.greycoprops(grey_co_matrices, "ASM")  # 或者energy energy = sqrt(ASM)
    contrast = GrayLCM.greycoprops(grey_co_matrices, "contrast")
    entropy = GrayLCM.get_entropy_of(grey_co_matrices)

    # 求均值
    energy_mean = np.mean(energy)
    contrast_mean = np.mean(contrast)
    entropy_mean = np.mean(entropy)

    # 求标准差
    energy_std = np.std(energy)
    contrast_std = np.std(contrast)
    entropy_std = np.std(entropy)
    return [energy_mean, contrast_mean, entropy_mean, energy_std, contrast_std, entropy_std]


if __name__ == '__main__':
    import test
    image = cv2.imread("../../TestImg/6.png")
    """"""

