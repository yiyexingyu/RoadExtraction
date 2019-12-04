# -*- encoding: utf-8 -*-
# @File    : FeatureExtraction1.py
# @Time    : 2019/11/26 16:19
# @Author  : 一叶星羽
# @Email   : h0670131005@gmail.com
# @Software: PyCharm

import cv2
import numpy as np
from numba import jit
from skimage.feature import greycomatrix


@jit
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


def adjust_gray_level(src_gray_img: np.ndarray, gray_level: int):
    """对图片进行指定灰度级范围内进行归一化处理"""
    assert src_gray_img.ndim == 2

    gray_img = src_gray_img.astype(np.int32)
    while gray_img.max() >= gray_level:
        gray_img = (gray_img / gray_level).astype(np.int32)  # type: np.ndarray
    return gray_img


@jit(forceobj=True)
def texture_vector(data):
    result = np.ones((data.shape[0], 6), dtype=np.float16)

    for i in range(data.shape[0]):
        grey_cm = data[i, :, :, 0, :]
        feature = np.ones((grey_cm.shape[2], 3), dtype=np.float16)

        for j in range(grey_cm.shape[2]):
            feature[j] = get_road_texture(grey_cm[:, :, j])

        energy, contract, entropy = feature[:, 0], feature[:, 1], feature[:, 2]
        result[i][0] = np.add.reduce(energy) / 4
        result[i][1] = np.add.reduce(contract) / 4
        result[i][2] = np.add.reduce(entropy) / 4

        result[i][3] = np.sqrt(np.add.reduce(np.square(energy - result[i][0])) / 4)
        result[i][4] = np.sqrt(np.add.reduce(np.square(contract - result[i][1])) / 4)
        result[i][5] = np.sqrt(np.add.reduce(np.square(entropy - result[i][2])) / 4)
    return result


def spectral_vector1(data):
    r = data[:, :, :, 0]
    g = data[:, :, :, 1]
    b = data[:, :, :, 2]
    length = data.shape[0]

    r_mean = np.apply_over_axes(np.mean, r, (1, 2)).reshape(length)
    g_mean = np.apply_over_axes(np.mean, g, (1, 2)).reshape(length)
    b_mean = np.apply_over_axes(np.mean, b, (1, 2)).reshape(length)

    r_std = np.apply_over_axes(np.std, r, (1, 2)).reshape(length)
    g_std = np.apply_over_axes(np.std, g, (1, 2)).reshape(length)
    b_std = np.apply_over_axes(np.std, b, (1, 2)).reshape(length)

    return np.vstack((r_mean, g_mean, b_mean, r_std, g_std, b_std))


def feature_extraction2(image: np.ndarray, level: int = 16, kernel: int = 5):
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
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    padding_num = kernel // 2
    height, width = image.shape[0], image.shape[1]

    rgb_img = np.ones((image.shape[0] + padding_num * 2, image.shape[1] + padding_num * 2, 3))
    rgb_img[..., 0] = np.pad(image[..., 0], padding_num, "mean")
    rgb_img[..., 1] = np.pad(image[..., 1], padding_num, "mean")
    rgb_img[..., 2] = np.pad(image[..., 2], padding_num, "mean")
    rgb_img = rgb_img.astype(np.uint8)
    grey_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    grey_img = adjust_gray_level(grey_img, gray_level=level)

    rgb_img = np.ascontiguousarray(rgb_img)
    grey_img = np.ascontiguousarray(grey_img)
    # rgb_img = rgb_img / 255

    rgb_pixels_slice = np.ones((height * width, kernel, kernel, 3))
    grey_pixels_slice = np.ones((height * width, level, level, 1, 4))
    # height, width = rgb_img.shape[0] - kernel + 1, rgb_img.shape[1] - kernel + 1

    for y in range(0, height):
        y_stop = y + kernel
        for x in range(0, width):
            x_stop = x + kernel
            index = y * width + x
            rgb_pixels_slice[index] = rgb_img[y: y_stop, x: x_stop, :]
            grey_pixels = grey_img[y: y_stop, x: x_stop]
            grey_pixels_slice[index] = greycomatrix(grey_pixels, [1], angles, level, normed=True)

    pixel_texture_vector = texture_vector(grey_pixels_slice)  # type: np.ndarray
    pixel_spectral_vector = spectral_vector1(rgb_pixels_slice).T
    result = np.hstack((pixel_texture_vector, pixel_spectral_vector))

    return result
